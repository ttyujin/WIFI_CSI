import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.analyze_activity_bimodality import (
    GROUP_HIGH,
    GROUP_LOW,
    GROUP_UNSEPARATED,
    AnalyzedFrame,
    GroupFit,
    SessionAnalysis,
    analyze_session,
    baseline_session_specs,
    build_window_rows,
    fit_frame_mean_groups,
    modulo_statistics,
    temporal_run_statistics,
    _profile_statistics,
)
from tools.prepare_activity_dataset import SessionSpec


def make_fit(groups):
    low_count = groups.count(GROUP_LOW)
    high_count = groups.count(GROUP_HIGH)
    return GroupFit(
        "SEPARATED", "test", tuple(groups), tuple(groups), low_count, high_count,
        1.0, 10.0, 9.0, 1.0, 20.0, min(low_count, high_count) / len(groups), 0.99,
    )


def make_analyzed_session(label, session_id, timestamps, groups):
    frames = [
        AnalyzedFrame(
            label, session_id, timestamp, index, index, -50.0, 1.0, 0.1,
            group, group, 0 if index else None, 0 if index else None,
            groups[index - 1] != group if index else None,
        )
        for index, (timestamp, group) in enumerate(zip(timestamps, groups))
    ]
    spec = SessionSpec(label, session_id, Path(f"{session_id}.jsonl"), f"{label}/{session_id}.jsonl")
    return SessionAnalysis(spec, make_fit(groups), frames, {}, [], len(frames), len(frames), 0, 0, 0, 0)


class AnalyzeActivityBimodalityTests(unittest.TestCase):
    def test_baseline_selection_is_exactly_24_sessions(self):
        specs = baseline_session_specs("activity")
        self.assertEqual(len(specs), 24)
        self.assertEqual(
            [spec.session_id for spec in specs],
            [f"{label}_{index:03d}" for label in ("sitting", "moving", "lying") for index in range(1, 9)],
        )

    def test_grouping_is_deterministic_and_rejects_unseparated_values(self):
        values = [1.0, 1.1, 0.9, 10.0, 10.1, 9.9] * 5
        first = fit_frame_mean_groups(values)
        second = fit_frame_mean_groups(values)
        self.assertEqual(first, second)
        self.assertEqual(first.status, "SEPARATED")
        weak = fit_frame_mean_groups([1.0 + index * 0.001 for index in range(30)])
        self.assertEqual(weak.status, "UNSEPARATED")
        self.assertEqual(set(weak.groups), {GROUP_UNSEPARATED})

    def test_temporal_runs(self):
        groups = [GROUP_LOW, GROUP_LOW, GROUP_HIGH, GROUP_HIGH, GROUP_LOW]
        summary = temporal_run_statistics(groups, [0.0, 1.0, 2.0, 3.0, 5.0])
        self.assertEqual(summary["group_transition_count"], 2)
        self.assertEqual(summary["run_count"], 3)
        self.assertAlmostEqual(summary["average_run_length_frames"], 5 / 3)
        self.assertEqual(summary["longest_run_frames"], 2)
        self.assertEqual(summary["first_group"], GROUP_LOW)
        self.assertEqual(summary["last_group"], GROUP_LOW)

    def test_sequence_modulo_association(self):
        groups = [GROUP_LOW, GROUP_HIGH] * 8
        values = list(range(16))
        mod2_v, detail = modulo_statistics(groups, values, 2)
        mod4_v, _ = modulo_statistics(groups, values, 4)
        self.assertAlmostEqual(mod2_v, 1.0)
        self.assertAlmostEqual(mod4_v, 1.0)
        self.assertEqual(json.loads(detail), {"LOW": [8, 0], "HIGH": [0, 8]})

    def test_frame_mean_normalization_comparison(self):
        from tools.analyze_activity_bimodality import RawFrame

        low_profile = [float(index + 1) for index in range(306)]
        high_profile = [value * 2.0 for value in low_profile]
        frames = []
        for index, amplitude in enumerate((low_profile, low_profile, high_profile, high_profile)):
            mean = sum(amplitude) / len(amplitude)
            std = (sum((value - mean) ** 2 for value in amplitude) / len(amplitude)) ** 0.5
            frames.append(RawFrame(float(index), index, index, -50.0, mean, std, tuple(amplitude)))
        fit = GroupFit(
            "SEPARATED", "test", (GROUP_LOW, GROUP_LOW, GROUP_HIGH, GROUP_HIGH),
            (GROUP_LOW, GROUP_LOW, GROUP_HIGH, GROUP_HIGH), 2, 2,
            153.5, 307.0, 153.5, 1.0, 20.0, 0.5, 0.99,
        )
        spec = SessionSpec("SITTING", "sitting_001", Path("x"), "SITTING/sitting_001.jsonl")
        summary, _ = _profile_statistics(spec, frames, fit)
        self.assertAlmostEqual(summary["mean_scale_ratio_high_over_low"], 2.0)
        self.assertAlmostEqual(summary["raw_profile_correlation"], 1.0)
        self.assertAlmostEqual(summary["raw_profile_scale_fit_relative_rmse"], 0.0)
        self.assertAlmostEqual(summary["normalized_profile_correlation"], 1.0)
        self.assertAlmostEqual(summary["normalized_profile_rmse"], 0.0)
        self.assertAlmostEqual(summary["normalized_subcarrier_fraction_abs_diff_gt_0_1"], 0.0)

    def test_windows_preserve_session_boundary(self):
        first = make_analyzed_session(
            "SITTING", "sitting_001", [0.0, 1.0, 2.0],
            [GROUP_LOW, GROUP_HIGH, GROUP_LOW],
        )
        second = make_analyzed_session(
            "MOVING", "moving_001", [0.0, 1.0, 2.0],
            [GROUP_HIGH, GROUP_HIGH, GROUP_HIGH],
        )
        manifests = [
            {"label": "SITTING", "session_id": "sitting_001", "window_index": "0", "group_id": "sitting_001", "start_timestamp": "0", "end_timestamp": "3"},
            {"label": "MOVING", "session_id": "moving_001", "window_index": "0", "group_id": "moving_001", "start_timestamp": "0", "end_timestamp": "3"},
        ]
        rows = build_window_rows(manifests, [first, second])
        self.assertEqual(rows[0]["low_frame_count"], 2)
        self.assertEqual(rows[1]["low_frame_count"], 0)
        self.assertEqual(rows[0]["diagnostic_frame_count"], 3)
        self.assertEqual(rows[1]["diagnostic_frame_count"], 3)
        invalid = [{**manifests[0], "group_id": "moving_001"}]
        with self.assertRaises(ValueError):
            build_window_rows(invalid, [first, second])

    def test_malformed_frame_is_excluded_safely(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "SITTING" / "sitting_001.jsonl"
            path.parent.mkdir(parents=True)
            metadata = {"record_type": "session", "label": "SITTING", "session_id": "sitting_001", "source": "esp32"}
            frame = {
                "record_type": "frame", "label": "SITTING", "session_id": "sitting_001",
                "source": "esp32", "timestamp": 1.0, "server_tick": 1, "sequence": 1,
                "node_id": 1, "rssi_dbm": -50.0, "subcarrier_count": 306,
                "amplitude_count": 306, "amplitude": [1.0] * 306,
            }
            with path.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(metadata) + "\n")
                stream.write("{broken\n")
                stream.write(json.dumps({**frame, "server_tick": None}) + "\n")
                stream.write(json.dumps(frame) + "\n")
            spec = SessionSpec("SITTING", "sitting_001", path, "SITTING/sitting_001.jsonl")
            analysis = analyze_session(spec)
            self.assertEqual(len(analysis.frames), 1)
            self.assertEqual(analysis.excluded_frames, 1)
            self.assertEqual(analysis.parse_errors, 1)
            self.assertEqual(analysis.summary["audit_status"], "WARN")


if __name__ == "__main__":
    unittest.main()
