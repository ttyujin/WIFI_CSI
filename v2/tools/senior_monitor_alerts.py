"""Local profile persistence and at-most-one SMTP attempt per interval stage.

Collection/service heuristics, NOT medical thresholds. No third-party packages.
SMTP delivery cannot provide exactly-once guarantees; uncertain sends are never
automatically retried. Credentials are read only from process environment.
"""
import json
import logging
import math
import os
import queue
import re
import smtplib
import ssl
import tempfile
import threading
from dataclasses import dataclass, field
from email.message import EmailMessage

LOG = logging.getLogger(__name__)
STAYING_NOTICE_SECONDS = 10800.0
MOVING_NOTICE_SECONDS = 3600.0
CAUTION_SECONDS = 14400.0
DANGER_SECONDS = 21600.0
# Compatibility aliases for code that imported the original notice constants.
STAYING_ALERT_SECONDS = STAYING_NOTICE_SECONDS
MOVING_ALERT_SECONDS = MOVING_NOTICE_SECONDS
DEFAULT_SENDER = "wifieldersender@gmail.com"


def valid_email(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 254 or not value.isascii():
        return False
    if not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+", value):
        return False
    local, domain = value.rsplit("@", 1)
    return (len(local) <= 64 and not local.startswith(".") and not local.endswith(".")
            and ".." not in local and "." in domain
            and all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", part)
                    for part in domain.split(".")))


def validate_profile(value):
    if not isinstance(value, dict) or set(value) != {"name", "gender", "guardian_email"}:
        raise ValueError("invalid_profile_fields")
    name, gender, email = value["name"], value["gender"], value["guardian_email"]
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 40 or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise ValueError("invalid_name")
    if gender not in ("female", "male"):
        raise ValueError("invalid_gender")
    if not isinstance(email, str) or not valid_email(email.strip()):
        raise ValueError("invalid_guardian_email")
    return {"name": name.strip(), "gender": gender, "guardian_email": email.strip()}


class ProfileStore:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.profile = None
        try:
            self.profile = validate_profile(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            pass
        except (OSError, ValueError, UnicodeError):
            LOG.warning("Profile unavailable: invalid or unreadable local profile; alerts disabled.")

    def get(self):
        with self.lock:
            return dict(self.profile) if self.profile else None

    def save(self, value):
        profile = validate_profile(value)
        # Separate profile lock; disk I/O never holds the activity state lock.
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=".profile-", suffix=".tmp", delete=False) as file:
                    temporary = file.name
                    json.dump(profile, file, ensure_ascii=False)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, self.path)
                self.profile = profile
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
        return dict(profile)


@dataclass(frozen=True)
class AlertConfig:
    sender: str
    password: str = field(repr=False)
    staying_seconds: float = STAYING_ALERT_SECONDS
    moving_seconds: float = MOVING_ALERT_SECONDS
    enabled: bool = False
    caution_seconds: float = CAUTION_SECONDS
    danger_seconds: float = DANGER_SECONDS

    def thresholds_for(self, state):
        notice = self.staying_seconds if state == "STAYING" else self.moving_seconds
        return (("NOTICE", notice), ("CAUTION", self.caution_seconds),
                ("DANGER", self.danger_seconds))

    @classmethod
    def from_environment(cls, env=None):
        env = os.environ if env is None else env
        sender = env.get("WIFI_ELDER_SENDER_EMAIL", DEFAULT_SENDER).strip()
        password = env.get("WIFI_ELDER_SENDER_APP_PASSWORD", "").strip()
        valid = True
        thresholds = []
        settings = (
            ("WIFI_ELDER_STAYING_NOTICE_SECONDS", "WIFI_ELDER_STAYING_ALERT_SECONDS",
             STAYING_NOTICE_SECONDS),
            ("WIFI_ELDER_MOVING_NOTICE_SECONDS", "WIFI_ELDER_MOVING_ALERT_SECONDS",
             MOVING_NOTICE_SECONDS),
            ("WIFI_ELDER_CAUTION_SECONDS", None, CAUTION_SECONDS),
            ("WIFI_ELDER_DANGER_SECONDS", None, DANGER_SECONDS),
        )
        for key, legacy_key, default in settings:
            try:
                raw = env.get(key, env.get(legacy_key, default) if legacy_key else default)
                seconds = float(raw)
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError
            except (ValueError, TypeError, OverflowError):
                valid = False
                seconds = default
            thresholds.append(seconds)
        staying, moving, caution, danger = thresholds
        if not (staying < caution < danger and moving < caution < danger):
            valid = False
        enabled = valid and valid_email(sender) and bool(password)
        if not enabled:
            # One startup warning, never credential values or exception messages.
            LOG.warning("Email alerts disabled: configure sender App Password and positive, ordered alert thresholds.")
        return cls(sender=sender, password=password, staying_seconds=staying,
                   moving_seconds=moving, enabled=enabled,
                   caution_seconds=caution, danger_seconds=danger)


def duration_text(seconds):
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}"


def create_message(config, event):
    profile = event["profile"]
    state = event["state"]
    duration = duration_text(event["confirmed_duration_seconds"])
    name = profile["name"]
    stage = event["stage"]
    label = "머무르는 중" if state == "STAYING" else "활동 중"
    continuing = stage != "NOTICE"
    observation = ("큰 움직임이 감지되지 않고 있습니다." if state == "STAYING" and continuing
                   else "큰 움직임이 감지되지 않았습니다." if state == "STAYING"
                   else "움직임 상태가 연속으로 감지되고 있습니다.")
    stage_label = {"NOTICE": "상태 확인", "CAUTION": "주의", "DANGER": "위험"}[stage]
    subject_tail = {"NOTICE": "생활 상태 안내", "CAUTION": "생활 상태 확인 필요",
                    "DANGER": "생활 상태 확인 요청"}[stage]
    message = EmailMessage()
    message["From"] = config.sender
    message["To"] = profile["guardian_email"]
    message["Subject"] = f"[WiFi Elder][{stage_label}] {name} 어르신 {subject_tail}"
    lines = ["안녕하세요.", "", f"{name} 어르신에게서", f"{duration} 동안 {observation}", ""]
    if stage == "NOTICE" and state == "MOVING":
        lines.extend(["평소 활동과 다른 상황인지 확인이 필요하다면", "연락하여 현재 상태를 확인해 주세요.", ""])
    elif stage == "DANGER":
        lines.extend(["설정된 '위험' 지속시간 기준에 도달했습니다.", "",
                      "가능한 경우 보호자가 직접 연락하여", "현재 상태를 확인해 주세요.", ""])
    elif stage == "CAUTION":
        lines.extend(["설정된 '주의' 지속시간 기준에 도달했습니다.", "",
                      "가능하시면 연락하여 현재 상태를 확인해 주세요.", ""])
    else:
        lines.extend(["가능하시면 연락하여 현재 상태를 확인해 주세요.", ""])
    lines.append(f"감지 상태: {label}")
    if stage != "NOTICE":
        lines.append(f"알림 단계: {stage_label}")
    lines.extend([f"지속 시간: {duration}", f"감지 시각: {event['last_prediction_at']}", ""])
    if stage == "NOTICE":
        lines.extend(["이 알림은 Wi-Fi CSI 기반 생활 상태 감지 결과를",
                      "바탕으로 자동 발송되었습니다."])
    else:
        lines.extend(["이 알림 단계는 지속시간에 따라 설정된 서비스 기준이며,",
                      "건강 상태나 응급 여부를 판단한 결과가 아닙니다."])
    message.set_content("\n".join(lines) + "\n")
    return message


class EmailAlerts:
    def __init__(self, config, profiles, current, smtp_factory=None):
        self.config, self.profiles, self.current = config, profiles, current
        self.smtp_factory = smtp_factory or smtplib.SMTP_SSL
        self.queue = queue.Queue(maxsize=64)
        self.guard = threading.Lock()
        self.claimed_interval = None
        self.claimed_stages = set()
        self.stopped = threading.Event()
        self.thread = None

    def enabled(self):
        return self.config.enabled and self.profiles.get() is not None

    def consider(self, snapshot):
        if not self.config.enabled or not snapshot["is_live"] or snapshot["state"] not in ("STAYING", "MOVING"):
            return
        profile = self.profiles.get()
        if profile is None:
            return
        interval = snapshot["interval_id"]
        duration = snapshot["confirmed_duration_seconds"]
        with self.guard:
            if self.claimed_interval is not None and interval < self.claimed_interval:
                return
            if interval != self.claimed_interval:
                self.claimed_interval = interval
                self.claimed_stages = set()
            for stage, limit in self.config.thresholds_for(snapshot["state"]):
                if duration < limit or stage in self.claimed_stages:
                    continue
                # Claim before enqueue/send. SMTP failures cannot re-arm this stage.
                self.claimed_stages.add(stage)
                try:
                    self.queue.put_nowait({**snapshot, "profile": profile, "stage": stage})
                except queue.Full:
                    LOG.warning("Email alert stage skipped: bounded queue full; no automatic retry.")

    def process_one(self, event):
        def still_eligible():
            current = self.current()
            return (self.config.enabled and current["is_live"]
                    and current["interval_id"] == event["interval_id"]
                    and current["state"] == event["state"]
                    and self.profiles.get() == event["profile"])
        try:
            if not still_eligible():
                return  # Cancel queued obsolete/disconnected/recipient-changed alerts.
            message = create_message(self.config, event)
            # No state/profile locks held during connect/login/send.
            with self.smtp_factory("smtp.gmail.com", 465, timeout=10,
                                   context=ssl.create_default_context()) as client:
                client.login(self.config.sender, self.config.password)
                # Connect/login may be slow. Recheck immediately before SMTP DATA.
                if still_eligible():
                    client.send_message(message)
        except Exception as error:
            # SMTP error strings may contain credentials or addresses. Log type only.
            LOG.warning("Email alert failed (%s); not retried to avoid duplicate delivery.", type(error).__name__)

    def _run(self):
        while not self.stopped.is_set():
            try:
                event = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.process_one(event)
            finally:
                self.queue.task_done()

    def start(self):
        if self.config.enabled and self.thread is None:
            self.thread = threading.Thread(target=self._run, name="email-alert-worker", daemon=True)
            self.thread.start()

    def stop(self):
        self.stopped.set()
