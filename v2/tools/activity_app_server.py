"""CSI disconnects are gaps, never activity intervals. Model/features unchanged."""

import json
import math
import time
import threading
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from senior_monitor_alerts import AlertConfig, EmailAlerts, ProfileStore


BASE_DIR = Path(__file__).resolve().parents[1]

WS_URL = "ws://localhost:3001/ws/activity/csi"

MODEL_PATH = (
    BASE_DIR
    / "data/models/activity/activity_heightLayoutB_v2.joblib"
)

WINDOW_SEC = 3.0
HOP_SEC = 1.5

MOVING_THRESHOLD = 0.70

EXPECTED_SUBCARRIERS = 306
CSI_TIMEOUT_SEC = 5.0

API_HOST = "0.0.0.0"
API_PORT = 8010


def iso(epoch):
    return (
        datetime
        .fromtimestamp(epoch)
        .astimezone()
        .isoformat(timespec="milliseconds")
    )


def finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False

    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def parse_csi(message):
    """Existing Rust activity_csi WS contract; no status/vitals messages."""

    try:
        data = json.loads(message)
    except (ValueError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    if data.get("type") != "activity_csi":
        return None

    amplitude = data.get("amplitude")
    timestamp = data.get("timestamp")
    sequence = data.get("sequence")

    # Bound timestamps to the representable calendar range.
    # Huge floats can otherwise make next_end += HOP_SEC
    # stop advancing and hang the window loop.
    if (
        not finite_number(timestamp)
        or not 0 <= timestamp <= 253402300799
        or type(sequence) is not int
        or not 0 <= sequence <= 0xFFFFFFFF
        or type(data.get("node_id")) is not int
        or not 1 <= data["node_id"] <= 255
        or data.get("subcarrier_count") != EXPECTED_SUBCARRIERS
        or data.get("amplitude_count") != EXPECTED_SUBCARRIERS
        or not isinstance(amplitude, list)
        or len(amplitude) != EXPECTED_SUBCARRIERS
        or not all(finite_number(value) for value in amplitude)
    ):
        return None

    return timestamp, sequence, amplitude


class StreamState:
    """
    One locked authority for freshness,
    intervals and atomic HTTP snapshots.
    """

    def __init__(
        self,
        wall=time.time,
        mono=time.monotonic,
        on_prediction=None,
    ):
        self.wall = wall
        self.mono = mono

        self.lock = threading.Lock()

        self.status = "CONNECTING"
        self.state = "UNKNOWN"

        self.started = None
        self.started_mono = None

        self.last_csi = None
        self.last_csi_mono = None

        self.last_prediction = None
        self.last_prediction_mono = None

        self.probability = None
        self.updated = None

        self.history = []

        self.generation = 0
        self.wait_since = mono()

        self.interval_id = 0

        self.on_prediction = on_prediction

    def _close(self, end):
        if (
            self.state != "UNKNOWN"
            and self.started is not None
        ):
            end = max(self.started, end)

            self.history.insert(
                0,
                {
                    "state": self.state,
                    "started_at": iso(self.started),
                    "ended_at": iso(end),
                    "duration_seconds": end - self.started,
                },
            )

    def _interrupt(self):
        if self.last_prediction is not None:
            self._close(self.last_prediction)

        self.state = "UNKNOWN"
        self.status = "RECONNECTING"

        self.started = None
        self.started_mono = None
        self.probability = None

        self.last_csi_mono = None

        self.updated = self.wall()

        self.generation += 1
        self.wait_since = self.mono()

    def _expire(self):
        # One freshness policy for HTTP reads
        # and receive-loop wakeups.
        now = self.mono()

        if self.last_csi_mono is not None:
            anchor = self.last_csi_mono
        else:
            anchor = self.wait_since

        prediction_stale = (
            self.status == "LIVE"
            and self.last_prediction_mono is not None
            and now - self.last_prediction_mono >= CSI_TIMEOUT_SEC
        )

        if (
            now - anchor >= CSI_TIMEOUT_SEC
            or prediction_stale
        ):
            self._interrupt()

    def interrupt(self):
        with self.lock:
            self._interrupt()

    def socket_connected(self):
        # A new socket gets a receive grace period,
        # never an activity or LIVE.
        with self.lock:
            self.wait_since = self.mono()

    def generation_now(self):
        with self.lock:
            self._expire()
            return self.generation

    def receive_valid(self):
        with self.lock:
            self._expire()

            self.last_csi = self.wall()
            self.last_csi_mono = self.mono()

            return self.generation

    def predict(self, probability, generation):
        if (
            not finite_number(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError(
                "Invalid MOVING probability"
            )

        with self.lock:
            self._expire()

            if (
                generation != self.generation
                or self.last_csi_mono is None
            ):
                # Discarded-window predictions
                # cannot revive LIVE.
                return False

            now = self.wall()
            mono = self.mono()

            if (
                self.last_prediction is not None
                and now < self.last_prediction
            ):
                self._interrupt()
                return False

            state = (
                "MOVING"
                if probability >= MOVING_THRESHOLD
                else "STAYING"
            )

            if state != self.state:
                self._close(now)

                self.started = now
                self.started_mono = mono

                self.interval_id += 1

            self.state = state
            self.status = "LIVE"

            self.probability = float(probability)

            self.last_prediction = now
            self.last_prediction_mono = mono

            self.updated = now

        if self.on_prediction is not None:
            self.on_prediction(
                self.snapshot()
            )

        return True

    def snapshot(self, include_history=False):
        with self.lock:
            self._expire()

            live = self.status == "LIVE"

            if live:
                confirmed = max(
                    0.0,
                    self.last_prediction - self.started,
                )
            else:
                confirmed = 0.0

            # One-hop display lag smooths the timer
            # without extrapolating beyond the last prediction.
            # No speculative activity seconds are counted.
            if live:
                display = min(
                    confirmed,
                    max(
                        0.0,
                        self.mono()
                        - self.started_mono
                        - HOP_SEC,
                    ),
                )
            else:
                display = 0.0

            result = {
                "state": self.state,
                "connection_status": self.status,
                "is_live": live,
                "interval_id": self.interval_id,
                "started_at": (
                    iso(self.started)
                    if live
                    else None
                ),
                "duration_seconds": display,
                "confirmed_duration_seconds": confirmed,
                "moving_probability": self.probability,
                "updated_at": (
                    iso(self.updated)
                    if self.updated is not None
                    else None
                ),
                "last_prediction_at": (
                    iso(self.last_prediction)
                    if self.last_prediction is not None
                    else None
                ),
                "last_csi_received_at": (
                    iso(self.last_csi)
                    if self.last_csi is not None
                    else None
                ),
            }

            if include_history:
                result["history"] = [
                    dict(row)
                    for row in self.history
                ]

            return result


class ActivityStream:
    """
    Timestamp windows;
    inference/network I/O never hold the state lock.
    """

    def __init__(
        self,
        state,
        predict_probability,
    ):
        self.state = state
        self.predict_probability = predict_probability

        self.frames = deque()

        self.generation = None

        self.last_timestamp = None
        self.next_end = None

    def feed(self, message):
        # Invalid/status traffic must expire silence too.
        self.state.generation_now()

        frame = parse_csi(message)

        if frame is None:
            return

        timestamp, _, _ = frame

        generation = self.state.generation_now()

        if (
            generation == self.generation
            and self.last_timestamp is not None
        ):
            if timestamp == self.last_timestamp:
                return

            if (
                timestamp < self.last_timestamp
                or timestamp - self.last_timestamp
                >= CSI_TIMEOUT_SEC
            ):
                self.state.interrupt()

        generation = self.state.receive_valid()

        if generation != self.generation:
            self.frames.clear()
            self.next_end = None
            self.generation = generation

        self.last_timestamp = timestamp

        self.frames.append(frame)

        if self.next_end is None:
            self.next_end = (
                timestamp + WINDOW_SEC
            )

        while (
            self.frames
            and self.frames[0][0]
            < timestamp - 2 * WINDOW_SEC
        ):
            self.frames.popleft()

        while timestamp >= self.next_end:
            window = [
                row
                for row in self.frames
                if (
                    self.next_end - WINDOW_SEC
                    <= row[0]
                    < self.next_end
                )
            ]

            if window:
                try:
                    probability = (
                        self.predict_probability(
                            [
                                row[2]
                                for row in window
                            ]
                        )
                    )

                    if not self.state.predict(
                        probability,
                        generation,
                    ):
                        return

                except Exception as error:
                    self.state.interrupt()

                    print(
                        "Activity prediction failed:",
                        type(error).__name__,
                        str(error),
                    )

                    return

            self.next_end += HOP_SEC


stream_state = StreamState()

profile_store = None
email_alerts = None


class ActivityHandler(
    BaseHTTPRequestHandler
):

    def send_json(
        self,
        data,
        status=200,
    ):
        body = json.dumps(
            data,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        if self.path == "/api/profile":
            origin = self.headers.get("Origin")

            if (
                origin
                and self.profile_origin_allowed()
            ):
                self.send_header(
                    "Access-Control-Allow-Origin",
                    origin,
                )

                self.send_header(
                    "Vary",
                    "Origin",
                )

        else:
            self.send_header(
                "Access-Control-Allow-Origin",
                "*",
            )

        self.send_header(
            "Cache-Control",
            "no-store",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/profile":
            if not self.profile_origin_allowed():
                self.send_json(
                    {
                        "success": False,
                        "error": "origin_not_allowed",
                    },
                    403,
                )
                return

            self.send_json(
                {
                    "profile": (
                        profile_store.get()
                        if profile_store
                        else None
                    ),
                    "email_alerts_enabled": (
                        email_alerts.enabled()
                        if email_alerts
                        else False
                    ),
                }
            )

        elif self.path in (
            "/api/activity/current",
            "/api/activity/current?include_history=1",
        ):
            self.send_json(
                stream_state.snapshot(
                    include_history=(
                        "?" in self.path
                    )
                )
            )

        elif self.path == "/api/activity/history":
            self.send_json(
                stream_state.snapshot(
                    include_history=True
                )["history"]
            )

        elif self.path == "/health":
            self.send_json(
                {
                    "status": "ok",
                    "model": (
                        "activity_heightLayoutB_v2"
                    ),
                    "moving_threshold": (
                        MOVING_THRESHOLD
                    ),
                }
            )

        else:
            self.send_response(404)
            self.end_headers()

    def profile_origin_allowed(self):
        # LAN UI is allowed,
        # arbitrary websites cannot overwrite
        # guardian config.
        #
        # This is CSRF/CORS protection,
        # not authentication for untrusted LAN peers.
        origin = self.headers.get("Origin")

        if origin is None:
            return True

        try:
            parsed = urlsplit(origin)

            host = urlsplit(
                "//" + self.headers.get(
                    "Host",
                    "",
                )
            ).hostname

            return (
                parsed.scheme == "http"
                and parsed.hostname == host
                and parsed.port == 8090
                and not parsed.username
                and not parsed.password
                and parsed.path == ""
            )

        except ValueError:
            return False

    def do_OPTIONS(self):
        if (
            self.path != "/api/profile"
            or not self.profile_origin_allowed()
        ):
            self.send_json(
                {
                    "success": False,
                    "error": "origin_not_allowed",
                },
                403,
            )
            return

        self.send_response(204)

        self.send_header(
            "Access-Control-Allow-Origin",
            self.headers.get(
                "Origin",
                "null",
            ),
        )

        self.send_header(
            "Vary",
            "Origin",
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )

        self.end_headers()

    def do_POST(self):
        if self.path != "/api/profile":
            self.send_json(
                {
                    "success": False,
                    "error": "not_found",
                },
                404,
            )
            return

        if not self.profile_origin_allowed():
            self.send_json(
                {
                    "success": False,
                    "error": "origin_not_allowed",
                },
                403,
            )
            return

        try:
            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            if (
                not 0 < length <= 4096
                or self.headers.get_content_type()
                != "application/json"
            ):
                raise ValueError(
                    "invalid_request"
                )

            value = json.loads(
                self.rfile.read(length)
                .decode("utf-8")
            )

            if profile_store is None:
                self.send_json(
                    {
                        "success": False,
                        "error": "profile_unavailable",
                    },
                    503,
                )
                return

            profile = (
                profile_store.save(value)
            )

        except (
            ValueError,
            UnicodeError,
        ):
            self.send_json(
                {
                    "success": False,
                    "error": "invalid_profile",
                },
                400,
            )
            return

        except OSError:
            self.send_json(
                {
                    "success": False,
                    "error": "profile_save_failed",
                },
                503,
            )
            return

        self.send_json(
            {
                "success": True,
                "profile": profile,
                "email_alerts_enabled": (
                    email_alerts.enabled()
                    if email_alerts
                    else False
                ),
            }
        )

    def log_message(
        self,
        format,
        *args,
    ):
        return


def run_api(server=None):
    server = (
        server
        or ThreadingHTTPServer(
            (API_HOST, API_PORT),
            ActivityHandler,
        )
    )

    print(
        f"Activity API: "
        f"http://localhost:{API_PORT}"
    )

    server.serve_forever()


def run_inference():
    # Lazy imports let interval/connection tests
    # run without loading a model.

    import joblib
    import numpy as np
    import websocket

    from validate_heightLayoutA_cv import (
        extract_normalized_features,
    )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded:",
        MODEL_PATH,
    )

    print(
        "Classes:",
        model.classes_,
        "Features:",
        model.n_features_in_,
    )

    print(
        f"Rule: MOVING if "
        f"P(MOVING) >= "
        f"{MOVING_THRESHOLD:.2f}"
    )

    def predict_probability(
        amplitudes
    ):
        feature, _ = (
            extract_normalized_features(
                np.asarray(
                    amplitudes,
                    dtype=np.float64,
                )
            )
        )

        probabilities = (
            model.predict_proba(
                feature.reshape(1, -1)
            )[0]
        )

        return float(
            dict(
                zip(
                    model.classes_,
                    probabilities,
                )
            )["MOVING"]
        )

    stream = ActivityStream(
        stream_state,
        predict_probability,
    )

    ws = None
    socket_generation = None

    def close_socket():
        # Transport cleanup failure
        # must not prevent the next
        # connection attempt.

        if ws is not None:
            try:
                ws.close()

            except (
                websocket.WebSocketException,
                OSError,
            ):
                pass

    try:
        while True:

            if ws is None:
                try:
                    # Allow longer time for the initial
                    # TCP/WebSocket connection.
                    ws = websocket.create_connection(
                        WS_URL,
                        timeout=5,
                    )

                    # Once connected, keep the original
                    # short receive timeout so freshness /
                    # disconnect detection remains responsive.
                    ws.settimeout(1)

                    stream_state.socket_connected()

                    socket_generation = (
                        stream_state.generation_now()
                    )

                    print(
                        "CSI socket connected; "
                        "waiting for a new valid window"
                    )

                except (
                    websocket.WebSocketException,
                    OSError,
                ):
                    stream_state.interrupt()

                    time.sleep(1)

                    continue

            try:
                message = ws.recv()

                if not message:
                    raise (
                        websocket
                        .WebSocketConnectionClosedException()
                    )

                stream.feed(message)

            except (
                websocket.WebSocketTimeoutException
            ):
                # A receive wakeup,
                # not another freshness threshold.
                stream_state.generation_now()

            except (
                websocket.WebSocketException,
                OSError,
            ):
                stream_state.interrupt()

                close_socket()

                ws = None

                time.sleep(1)

            # HTTP reads can also expire CSI.
            # A generation change is the same
            # central interruption signal,
            # including a silent half-open socket.
            if (
                ws is not None
                and stream_state.generation_now()
                != socket_generation
            ):
                # A late frame may have arrived
                # after HTTP detected the gap.
                # Do not carry that old socket's
                # partial window into replacement.
                stream_state.interrupt()

                close_socket()

                ws = None

    finally:
        stream_state.interrupt()

        close_socket()


if __name__ == "__main__":
    # Bind before starting mail/inference:
    # a second process on the same port
    # must not silently run another sender
    # after its API thread fails to bind.

    api_server = ThreadingHTTPServer(
        (API_HOST, API_PORT),
        ActivityHandler,
    )

    profile_store = ProfileStore(
        BASE_DIR
        / "data/senior-monitor/profile.json"
    )

    email_alerts = EmailAlerts(
        AlertConfig.from_environment(),
        profile_store,
        stream_state.snapshot,
    )

    stream_state.on_prediction = (
        email_alerts.consider
    )

    email_alerts.start()

    threading.Thread(
        target=run_api,
        args=(api_server,),
        daemon=True,
    ).start()

    try:
        run_inference()

    except KeyboardInterrupt:
        print(
            "Activity server stopped."
        )

    finally:
        email_alerts.stop()

        api_server.shutdown()

        api_server.server_close()