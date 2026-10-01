"""No Gmail/network delivery: all SMTP clients are mocks, files use temp dirs."""
import json
import logging
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
import uuid

import activity_app_server as app
from senior_monitor_alerts import (AlertConfig, CAUTION_SECONDS, DANGER_SECONDS,
    EmailAlerts, MOVING_NOTICE_SECONDS, ProfileStore, STAYING_NOTICE_SECONDS,
    create_message, validate_profile)
from test_activity_app_server import Clock, frame

PROFILE = {"name": "홍길동", "gender": "female", "guardian_email": "guardian@example.com"}


class AlertTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "profile.json"
        self.profiles = ProfileStore(self.path)
        self.profiles.save(PROFILE)
        self.clock = Clock()
        self.state = app.StreamState(self.clock, self.clock)
        self.secret = uuid.uuid4().hex  # Runtime-generated fake, never a real credential.
        self.config = AlertConfig("wifieldersender@gmail.com", self.secret, 3, 3,
                                  True, 4, 6)
        self.smtp = MagicMock()
        self.client = self.smtp.return_value.__enter__.return_value
        self.alerts = EmailAlerts(self.config, self.profiles, self.state.snapshot, self.smtp)
        self.state.on_prediction = self.alerts.consider

    def prediction(self, probability=0.2, advance=0):
        self.clock.advance(advance)
        self.state.predict(probability, self.state.receive_valid())

    def drain(self):
        while not self.alerts.queue.empty():
            self.alerts.process_one(self.alerts.queue.get_nowait())

    def use_config(self, staying, moving, caution, danger):
        self.config = AlertConfig("wifieldersender@gmail.com", self.secret,
                                  staying, moving, True, caution, danger)
        self.alerts = EmailAlerts(self.config, self.profiles, self.state.snapshot, self.smtp)
        self.state.on_prediction = self.alerts.consider

    def assert_next_stage(self, expected):
        self.assertEqual(self.alerts.queue.qsize(), 1)
        event = self.alerts.queue.get_nowait()
        self.assertEqual(event["stage"], expected)
        self.alerts.process_one(event)

    def use_synthetic_live_interval(self, state):
        self.synthetic = {"is_live": True, "state": state, "interval_id": 15,
                          "confirmed_duration_seconds": 0,
                          "last_prediction_at": "2026-09-19T12:00:00+09:00"}
        config = AlertConfig("wifieldersender@gmail.com", self.secret,
                             STAYING_NOTICE_SECONDS, MOVING_NOTICE_SECONDS, True,
                             CAUTION_SECONDS, DANGER_SECONDS)
        self.alerts = EmailAlerts(config, self.profiles, lambda: dict(self.synthetic), self.smtp)

    def consider_duration(self, seconds):
        self.synthetic["confirmed_duration_seconds"] = seconds
        self.alerts.consider(dict(self.synthetic))

    def test_staying_exact_stage_boundaries_and_no_duplicates(self):
        self.use_synthetic_live_interval("STAYING")
        self.consider_duration(10799)
        self.assertTrue(self.alerts.queue.empty())                 # 1
        self.consider_duration(10800)
        self.assert_next_stage("NOTICE")                          # 2
        self.consider_duration(10801)
        self.assertTrue(self.alerts.queue.empty())                 # 3
        self.consider_duration(14399)
        self.assertTrue(self.alerts.queue.empty())                 # 4: 14399
        self.consider_duration(14400)
        self.assert_next_stage("CAUTION")                         # 5
        self.consider_duration(14401)
        self.assertTrue(self.alerts.queue.empty())                 # 6
        self.consider_duration(21600)
        self.assert_next_stage("DANGER")                          # 7: 21600
        self.consider_duration(25200)
        self.assertTrue(self.alerts.queue.empty())                 # 8
        self.assertEqual(self.client.send_message.call_count, 3)

    def test_moving_exact_stage_boundaries_and_no_duplicates(self):
        self.use_synthetic_live_interval("MOVING")
        self.consider_duration(3599)
        self.assertTrue(self.alerts.queue.empty())                 # 9
        self.consider_duration(3600)
        self.assert_next_stage("NOTICE")                          # 10
        self.consider_duration(3601)
        self.assertTrue(self.alerts.queue.empty())                 # 11
        self.consider_duration(14400)
        self.assert_next_stage("CAUTION")                         # 12: 14400
        self.consider_duration(14401)
        self.assertTrue(self.alerts.queue.empty())                 # 13
        self.consider_duration(21600)
        self.assert_next_stage("DANGER")                          # 14: 21600
        self.consider_duration(25200)
        self.assertTrue(self.alerts.queue.empty())                 # 15
        self.assertEqual(self.client.send_message.call_count, 3)

    def test_transition_rearms_same_state(self):
        interval_ids = []
        for probability in (0.2, 0.8, 0.2):
            self.prediction(probability)
            interval_ids.append(self.state.snapshot()["interval_id"])
            self.prediction(probability, 3)
            self.assertEqual(self.alerts.queue.queue[0]["stage"], "NOTICE")
            self.drain()
        self.assertEqual(self.client.send_message.call_count, 3)
        self.assertEqual(interval_ids, sorted(set(interval_ids)))  # 16-18

    def test_disconnect_gap_and_reconnect_window_excluded(self):
        stream = app.ActivityStream(self.state, lambda values: 0.2)
        def feed(delta):
            self.clock.advance(delta)
            stream.feed(frame(self.clock()))
        feed(0)
        feed(3)  # first prediction, duration zero
        feed(1.5)
        self.clock.advance(20)
        self.alerts.consider(self.state.snapshot())
        self.drain()
        self.smtp.assert_not_called()
        self.state.socket_connected()
        feed(0)
        feed(2.9)
        self.assertFalse(self.state.snapshot()["is_live"])
        feed(0.1)
        self.assertTrue(self.alerts.queue.empty())
        feed(1.5)
        feed(1.5)
        self.drain()
        self.client.send_message.assert_called_once()
        self.assertEqual(self.state.snapshot()["confirmed_duration_seconds"], 3)

    def test_connecting_unknown_reconnecting_invalid_csi_do_not_send(self):
        self.alerts.consider(self.state.snapshot())
        stream = app.ActivityStream(self.state, lambda values: 0.8)
        for _ in range(10):
            self.clock.advance(1)
            stream.feed(frame(self.clock(), subcarrier_count=0))
            self.alerts.consider(self.state.snapshot())
        self.assertEqual(self.state.snapshot()["connection_status"], "RECONNECTING")
        self.drain()
        self.smtp.assert_not_called()

    def test_no_profile_or_invalid_persisted_profile_disables_alerts(self):
        for value in (None, {**PROFILE, "guardian_email": ""}, {**PROFILE, "guardian_email": "bad"}):
            self.path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertLogs("senior_monitor_alerts", logging.WARNING):
                profiles = ProfileStore(self.path)
            alerts = EmailAlerts(self.config, profiles, self.state.snapshot, self.smtp)
            self.prediction()
            self.prediction(advance=3)
            alerts.consider(self.state.snapshot())
            self.assertFalse(alerts.enabled())
            self.assertTrue(alerts.queue.empty())
        self.smtp.assert_not_called()

    def test_missing_credentials_and_invalid_threshold_warn_once_without_secrets(self):
        for env in ({}, {"WIFI_ELDER_STAYING_NOTICE_SECONDS": "nan"},
                    {"WIFI_ELDER_MOVING_NOTICE_SECONDS": "0"},
                    {"WIFI_ELDER_CAUTION_SECONDS": "21600"},
                    {"WIFI_ELDER_DANGER_SECONDS": "14400"}):
            with self.assertLogs("senior_monitor_alerts", logging.WARNING) as logs:
                config = AlertConfig.from_environment(env)
            self.assertFalse(config.enabled)
            self.assertEqual(len(logs.output), 1)
            self.assertNotIn(self.secret, repr(config))
            alerts = EmailAlerts(config, self.profiles, self.state.snapshot, self.smtp)
            alerts.start()
            self.assertIsNone(alerts.thread)
        enabled = AlertConfig.from_environment({"WIFI_ELDER_SENDER_APP_PASSWORD": self.secret,
            "WIFI_ELDER_MOVING_NOTICE_SECONDS": "12", "WIFI_ELDER_STAYING_NOTICE_SECONDS": "20",
            "WIFI_ELDER_CAUTION_SECONDS": "40", "WIFI_ELDER_DANGER_SECONDS": "60"})
        self.assertTrue(enabled.enabled)
        self.assertEqual(enabled.staying_seconds, 20)
        self.assertEqual(enabled.moving_seconds, 12)
        self.assertEqual(enabled.caution_seconds, 40)
        self.assertEqual(enabled.danger_seconds, 60)
        self.assertNotIn(self.secret, repr(enabled))
        defaults = AlertConfig.from_environment({"WIFI_ELDER_SENDER_APP_PASSWORD": self.secret})
        self.assertTrue(defaults.enabled)
        self.assertEqual((defaults.staying_seconds, defaults.moving_seconds,
                          defaults.caution_seconds, defaults.danger_seconds),
                         (10800, 3600, 14400, 21600))
        legacy = AlertConfig.from_environment({"WIFI_ELDER_SENDER_APP_PASSWORD": self.secret,
            "WIFI_ELDER_MOVING_ALERT_SECONDS": "12", "WIFI_ELDER_STAYING_ALERT_SECONDS": "20"})
        self.assertTrue(legacy.enabled)
        self.assertEqual((legacy.staying_seconds, legacy.moving_seconds), (20, 12))

    def test_smtp_failure_is_safe_no_retry_or_credential_log(self):
        self.use_config(3, 3, 100, 200)
        self.client.send_message.side_effect = OSError(self.secret)
        self.prediction()
        self.prediction(advance=3)
        with self.assertLogs("senior_monitor_alerts", logging.WARNING) as logs:
            self.drain()
        self.assertNotIn(self.secret, "\n".join(logs.output))
        for _ in range(10):
            self.prediction(advance=1.5)
            self.drain()
        self.client.send_message.assert_called_once()
        self.assertTrue(self.state.snapshot()["is_live"])

    def test_queued_disconnected_or_recipient_changed_alert_is_cancelled(self):
        self.prediction()
        self.prediction(advance=3)
        self.state.interrupt()
        self.drain()
        self.smtp.assert_not_called()
        self.prediction()
        self.prediction(advance=3)
        self.profiles.save({**PROFILE, "guardian_email": "new@example.com"})
        self.drain()
        self.smtp.assert_not_called()

    def test_reconnect_cancels_old_interval_and_starts_new_stage_set(self):
        self.prediction()
        self.prediction(advance=3)
        old_interval = self.state.snapshot()["interval_id"]
        self.state.interrupt()
        self.state.socket_connected()
        self.drain()
        self.smtp.assert_not_called()                              # 20
        self.prediction()
        self.prediction(advance=3)
        self.assertGreater(self.state.snapshot()["interval_id"], old_interval)
        self.assertEqual(self.alerts.queue.queue[0]["stage"], "NOTICE")
        self.drain()
        self.client.send_message.assert_called_once()

    def test_mail_content_tls_and_network_outside_activity_lock(self):
        self.prediction(0.8)
        self.prediction(0.8, 3)
        self.client.send_message.side_effect = lambda message: self.state.snapshot()
        self.drain()  # callback reacquires lock, must not deadlock.
        args, kwargs = self.smtp.call_args
        self.assertEqual(args, ("smtp.gmail.com", 465))
        self.assertEqual(kwargs["timeout"], 10)
        message = self.client.send_message.call_args.args[0]
        self.assertEqual(message["To"], PROFILE["guardian_email"])
        self.assertEqual(message["Subject"], "[WiFi Elder][상태 확인] 홍길동 어르신 생활 상태 안내")
        self.assertIn("00:00:03", message.get_content())
        self.assertIn("활동 중", message.get_content())
        self.assertIn("평소 활동과 다른 상황인지", message.get_content())
        self.assertNotIn(self.secret, message.as_string())

    def test_caution_and_danger_content_are_duration_levels_not_medical_judgments(self):
        base = {"profile": PROFILE, "state": "STAYING", "last_prediction_at": "2026-09-19T12:00:00+09:00"}
        caution = create_message(self.config, {**base, "stage": "CAUTION",
                                  "confirmed_duration_seconds": 14400})
        danger = create_message(self.config, {**base, "stage": "DANGER",
                                 "confirmed_duration_seconds": 21600})
        notice = create_message(self.config, {**base, "stage": "NOTICE",
                                 "confirmed_duration_seconds": 10800})
        self.assertIn("큰 움직임이 감지되지 않았습니다", notice.get_content())
        self.assertEqual(caution["Subject"], "[WiFi Elder][주의] 홍길동 어르신 생활 상태 확인 필요")
        self.assertEqual(danger["Subject"], "[WiFi Elder][위험] 홍길동 어르신 생활 상태 확인 요청")
        for message, stage in ((caution, "주의"), (danger, "위험")):
            body = message.get_content()
            self.assertIn("알림 단계: " + stage, body)
            self.assertIn("지속시간에 따라 설정된 서비스 기준", body)
            self.assertIn("건강 상태나 응급 여부를 판단한 결과가 아닙니다", body)

    def test_disconnect_during_smtp_login_cancels_data_submission(self):
        self.prediction()
        self.prediction(advance=3)
        self.client.login.side_effect = lambda *args: self.state.interrupt()
        self.drain()
        self.client.send_message.assert_not_called()

    def test_dedicated_worker_does_not_block_prediction(self):
        entered, release = threading.Event(), threading.Event()
        def slow_send(message):
            entered.set()
            release.wait(2)
        self.client.send_message.side_effect = slow_send
        self.alerts.start()
        try:
            self.prediction()
            self.prediction(advance=3)
            self.assertTrue(entered.wait(1))
            self.prediction(advance=1.5)
            self.assertTrue(self.state.snapshot()["is_live"])
        finally:
            release.set()
            self.alerts.stop()
            self.alerts.thread.join(2)

    def test_atomic_persistence_and_save_failure_preserves_old_profile(self):
        self.assertEqual(ProfileStore(self.path).get(), PROFILE)
        with patch("senior_monitor_alerts.os.replace", side_effect=OSError):
            with self.assertRaises(OSError):
                self.profiles.save({**PROFILE, "name": "다른이름"})
        self.assertEqual(self.profiles.get(), PROFILE)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), PROFILE)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_profile_validation(self):
        self.assertEqual(validate_profile({**PROFILE, "name": " 홍길동 ", "guardian_email": " guardian@example.com "}), PROFILE)
        for value in ({}, None, {**PROFILE, "gender": "other"}, {**PROFILE, "name": ""},
                      {**PROFILE, "name": "a" * 41}, {**PROFILE, "name": "x\nBcc: y"},
                      {**PROFILE, "guardian_email": "x@example.com\nBcc:z@example.com"},
                      {**PROFILE, "guardian_email": ""}, {**PROFILE, "guardian_email": "a@-bad.com"},
                      {**PROFILE, "password": self.secret}):
            with self.assertRaises(ValueError):
                validate_profile(value)

    def test_profile_http_validation_cors_and_no_secret_disclosure(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), app.ActivityHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(app, "profile_store", self.profiles), patch.object(app, "email_alerts", self.alerts):
                base = "http://127.0.0.1:" + str(server.server_port)
                def post(value, origin="http://127.0.0.1:8090"):
                    return urlopen(Request(base + "/api/profile", method="POST",
                        data=json.dumps(value).encode(), headers={"Content-Type": "application/json", "Origin": origin}), timeout=2)
                with post({**PROFILE, "gender": "male"}) as response:
                    result = json.load(response)
                self.assertTrue(result["success"])
                with urlopen(base + "/api/profile", timeout=2) as response:
                    result = json.load(response)
                self.assertEqual(result["profile"]["gender"], "male")
                self.assertNotIn(self.secret, json.dumps(result))
                for value in ({**PROFILE, "guardian_email": "bad"}, {**PROFILE, "guardian_email": ""}):
                    with self.assertRaises(HTTPError) as error:
                        post(value)
                    self.assertEqual(error.exception.code, 400)
                with self.assertRaises(HTTPError) as error:
                    post(PROFILE, "http://untrusted.example:8090")
                self.assertEqual(error.exception.code, 403)
                with urlopen(Request(base + "/api/profile", method="OPTIONS",
                    headers={"Origin": "http://127.0.0.1:8090"}), timeout=2) as response:
                    self.assertEqual(response.status, 204)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_lan_and_model_contract_constants_unchanged(self):
        self.assertEqual(app.API_HOST, "0.0.0.0")
        self.assertEqual(app.API_PORT, 8010)
        self.assertEqual(app.WS_URL, "ws://localhost:3001/ws/activity/csi")
        self.assertEqual((app.MOVING_THRESHOLD, app.WINDOW_SEC, app.HOP_SEC,
                          app.EXPECTED_SUBCARRIERS), (0.70, 3.0, 1.5, 306))


if __name__ == "__main__":
    unittest.main()
