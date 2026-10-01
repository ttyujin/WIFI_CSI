"""SYNTHETIC deterministic connection tests; no model/data writes or hardware."""
import json
import threading
import sys
import unittest
from datetime import datetime
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.request import urlopen
from types import SimpleNamespace

import activity_app_server as app


class Clock:
    def __init__(self):
        self.now = 1789000000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def frame(ts, **overrides):
    value = dict(type="activity_csi", timestamp=ts, sequence=int(ts * 30) % (2 ** 32),
                 node_id=1, subcarrier_count=306, amplitude_count=306, amplitude=[1.0] * 306)
    value.update(overrides)
    return json.dumps(value)


class ActivityTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.state = app.StreamState(self.clock, self.clock)
        self.windows = []
        self.probability = 0.2
        def predict(amplitudes):
            self.windows.append(amplitudes)
            return self.probability
        self.stream = app.ActivityStream(self.state, predict)

    def feed(self, seconds=0, **overrides):
        self.clock.advance(seconds)
        self.stream.feed(frame(self.clock(), **overrides))

    def live(self, probability=0.2):
        self.probability = probability
        self.feed()
        for _ in range(6):
            self.feed(0.5)
        self.assertTrue(self.state.snapshot()["is_live"])

    def test_connect_and_valid_frame_are_not_live_without_full_window(self):
        self.assertEqual(self.state.snapshot()["connection_status"], "CONNECTING")
        self.feed()
        self.feed(2.9)
        self.assertFalse(self.state.snapshot()["is_live"])
        self.feed(0.1)
        self.assertEqual(self.state.snapshot()["state"], "STAYING")
        self.assertEqual(len(self.windows), 1)

    def test_both_live_states_grow_but_never_past_last_prediction(self):
        for probability, state in [(0.2, "STAYING"), (0.8, "MOVING")]:
            with self.subTest(state=state):
                self.setUp()
                self.live(probability)
                for _ in range(6):
                    self.feed(0.5)
                result = self.state.snapshot()
                self.assertEqual(result["state"], state)
                self.assertEqual(result["duration_seconds"], 1.5)
                self.assertEqual(result["confirmed_duration_seconds"], 3)
                self.clock.advance(2)
                self.assertEqual(self.state.snapshot()["duration_seconds"], 3)

    def test_both_disconnects_close_at_prediction_not_detection_and_exclude_gap(self):
        for probability in (0.2, 0.8):
            with self.subTest(probability=probability):
                self.setUp()
                self.live(probability)
                self.feed(1.5)
                last = self.state.snapshot()["last_prediction_at"]
                self.clock.advance(5)
                snapshot = self.state.snapshot(True)
                self.assertEqual(snapshot["connection_status"], "RECONNECTING")
                self.assertEqual(snapshot["state"], "UNKNOWN")
                self.assertEqual(snapshot["duration_seconds"], 0)
                self.assertEqual(snapshot["history"][0]["ended_at"], last)
                self.assertEqual(snapshot["history"][0]["duration_seconds"], 1.5)
                self.clock.advance(20)
                self.assertEqual(self.state.snapshot(True)["history"], snapshot["history"])
                self.feed()
                self.feed(2.9)
                self.assertFalse(self.state.snapshot()["is_live"])
                self.feed(0.1)
                recovered = self.state.snapshot(True)
                self.assertTrue(recovered["is_live"])
                self.assertEqual(recovered["duration_seconds"], 0)
                self.assertGreater(recovered["started_at"], last)
                self.assertEqual(len(recovered["history"]), 1)

    def test_invalid_zero_subcarriers_status_and_duplicate_traffic_do_not_refresh(self):
        for overrides in ({"subcarrier_count": 0}, {"type": "vitals"}, {"amplitude": []}):
            with self.subTest(overrides=overrides):
                self.setUp()
                self.live()
                last = self.state.last_csi_mono
                for _ in range(6):
                    self.feed(1, **overrides)
                self.assertFalse(self.state.snapshot()["is_live"])
                self.assertEqual(self.state.last_csi, last)
        self.setUp()
        self.live()
        timestamp = self.clock()
        for _ in range(6):
            self.clock.advance(1)
            self.stream.feed(frame(timestamp))
        self.assertFalse(self.state.snapshot()["is_live"])

    def test_timeout_boundary_is_five_seconds(self):
        self.live()
        self.clock.advance(4.999)
        self.assertTrue(self.state.snapshot()["is_live"])
        self.clock.advance(0.001)
        self.assertFalse(self.state.snapshot()["is_live"])

    def test_inference_does_not_hold_lock_and_stalled_prediction_expires(self):
        self.live()
        self.stream.predict_probability = lambda amplitudes: (
            self.state.snapshot()["moving_probability"])
        self.feed(1.5)  # snapshot takes the same lock; no inference lock/deadlock.
        for _ in range(6):
            self.clock.advance(1)
            self.state.receive_valid()  # Valid CSI alone is not successful inference.
        self.assertFalse(self.state.snapshot()["is_live"])

    def test_snapshot_history_is_detached_from_internal_state(self):
        self.live()
        self.state.interrupt()
        result = self.state.snapshot(True)
        result["history"][0]["state"] = "UNKNOWN"
        self.assertEqual(self.state.history[0]["state"], "STAYING")

    def test_late_prediction_from_discarded_window_is_rejected(self):
        self.live()
        generation = self.state.generation_now()
        self.state.interrupt()
        self.state.receive_valid()
        self.assertFalse(self.state.predict(0.8, generation))
        self.assertFalse(self.state.snapshot()["is_live"])

    def test_silent_reopened_socket_expires_again_without_faking_live(self):
        self.live()
        self.state.interrupt()
        for _ in range(3):
            self.state.socket_connected()
            generation = self.state.generation_now()
            self.assertEqual(self.state.snapshot()["connection_status"], "RECONNECTING")
            self.clock.advance(5)
            self.assertNotEqual(self.state.generation_now(), generation)
            self.assertFalse(self.state.snapshot()["is_live"])
        self.assertEqual(len(self.state.history), 1)

    def test_invalid_probability_interrupts_without_killing_receive_path(self):
        self.live()
        self.probability = float("nan")
        self.feed(1.5)
        self.assertFalse(self.state.snapshot()["is_live"])
        self.probability = 0.8
        self.feed(0.1)
        self.feed(3)
        self.assertTrue(self.state.snapshot()["is_live"])

    def test_timestamp_rollback_and_long_jump_require_new_window(self):
        for offset in (-10, 10):
            with self.subTest(offset=offset):
                self.setUp()
                self.live()
                self.stream.feed(frame(self.clock() + offset))
                self.assertFalse(self.state.snapshot()["is_live"])
                self.assertEqual(len(self.windows), 1)

    def test_malformed_frames_are_safe(self):
        values = ["bad", "null", "[]", "{}", frame(1000, amplitude=[None] * 306),
                  frame(1000, amplitude=[float("nan")] * 306), frame(1000, amplitude=["1"] * 306),
                  frame(1000, amplitude_count=0), frame(1000, node_id=0),
                  frame(1000, timestamp=float("nan")), frame(1000, sequence=True),
                  frame(1000, timestamp=1e308), frame(1000, amplitude=[10 ** 400] * 306)]
        for value in values:
            self.assertIsNone(app.parse_csi(value))
            self.stream.feed(value)
        self.assertIsNone(self.state.last_csi_mono)

    def test_threshold_070_exactly(self):
        self.assertEqual(app.MOVING_THRESHOLD, 0.70)
        for probability, expected in [(0.699999, "STAYING"), (0.700000, "MOVING")]:
            self.state.predict(probability, self.state.receive_valid())
            self.assertEqual(self.state.snapshot()["state"], expected)
        self.assertTrue(all(row["state"] in ("STAYING", "MOVING") for row in self.state.history))

    def test_midnight_disconnect_gap_has_no_interval(self):
        self.clock.now = datetime(2026, 9, 9, 23, 59, 55).astimezone().timestamp()
        self.live()
        self.feed(1.5)
        self.clock.advance(20)
        self.state.snapshot()
        self.live(0.8)
        snapshot = self.state.snapshot(True)
        end = datetime.fromisoformat(snapshot["history"][0]["ended_at"])
        start = datetime.fromisoformat(snapshot["started_at"])
        self.assertEqual(end.day, 9)
        self.assertEqual(start.day, 10)
        self.assertGreater((start - end).total_seconds(), 20)

    def test_http_200_can_explicitly_report_disconnected_atomic_snapshot(self):
        self.live()
        self.feed(1.5)
        server = ThreadingHTTPServer(("127.0.0.1", 0), app.ActivityHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(app, "stream_state", self.state):
                self.clock.advance(5)
                base = "http://127.0.0.1:" + str(server.server_port)
                with urlopen(base + "/api/activity/current?include_history=1", timeout=2) as response:
                    self.assertEqual(response.status, 200)
                    result = json.load(response)
                self.assertFalse(result["is_live"])
                self.assertEqual(result["state"], "UNKNOWN")
                self.assertEqual(len(result["history"]), 1)
                with urlopen(base + "/api/activity/history", timeout=2) as response:
                    self.assertEqual(json.load(response), result["history"])
                with urlopen(base + "/api/activity/current", timeout=2) as response:
                    self.assertNotIn("history", json.load(response))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_actual_receive_loop_reconnects_silent_socket_and_requires_new_prediction(self):
        # Exercise run_inference itself, not a parallel implementation. Model,
        # transport and clock are explicit SYNTHETIC test doubles.
        snapshots = []
        closed = []
        connect_timeouts = []
        receive_timeouts = []
        clock, state = self.clock, self.state
        class SocketError(Exception):
            pass
        class Timeout(SocketError):
            pass
        class Socket:
            def __init__(self, name, events):
                self.name, self.events = name, iter(events)
            def recv(self):
                snapshots.append(state.snapshot(True))
                advance, kind = next(self.events, (0, "stop"))
                clock.advance(advance)
                if kind == "stop":
                    raise KeyboardInterrupt
                if kind == "timeout":
                    raise Timeout
                return frame(clock())
            def settimeout(self, seconds):
                receive_timeouts.append(seconds)
            def close(self):
                closed.append(self.name)
                if self.name == "first":
                    raise OSError("synthetic close failure")
        sockets = iter([
            Socket("first", [(0, "frame")] + [(0.5, "frame")] * 9 + [(1, "timeout")] * 5),
            Socket("second", [(0, "frame")] + [(0.5, "frame")] * 7),
        ])
        model = SimpleNamespace(classes_=["LYING", "MOVING", "SITTING"],
                                n_features_in_=918, predict_proba=lambda features: [[0.1, 0.8, 0.1]])
        extracted = []
        def extract(amplitude):
            self.assertTrue(all(len(row) == 306 for row in amplitude))
            extracted.append(amplitude)
            return SimpleNamespace(reshape=lambda *args: "normalized-features"), 0
        def create_connection(*args, **kwargs):
            connect_timeouts.append(kwargs.get("timeout"))
            return next(sockets)
        modules = {
            "joblib": SimpleNamespace(load=lambda path: model),
            "numpy": SimpleNamespace(asarray=lambda values, **kwargs: values, float64=float),
            "websocket": SimpleNamespace(create_connection=create_connection,
                WebSocketException=SocketError, WebSocketTimeoutException=Timeout,
                WebSocketConnectionClosedException=SocketError),
            "validate_heightLayoutA_cv": SimpleNamespace(extract_normalized_features=extract),
        }
        with patch.dict(sys.modules, modules), patch.object(app, "stream_state", state):
            with self.assertRaises(KeyboardInterrupt):
                app.run_inference()
        self.assertEqual(closed, ["first", "second"])
        self.assertEqual(connect_timeouts, [5, 5])
        self.assertEqual(receive_timeouts, [1, 1])
        self.assertTrue(extracted)
        self.assertTrue(any(s["connection_status"] == "RECONNECTING" for s in snapshots))
        starts = {s["started_at"] for s in snapshots if s["is_live"]}
        self.assertEqual(len(starts), 2)
        records = state.snapshot(True)["history"]
        self.assertEqual(len(records), 2)
        self.assertGreater(records[0]["started_at"], records[1]["ended_at"])


if __name__ == "__main__":
    unittest.main()
