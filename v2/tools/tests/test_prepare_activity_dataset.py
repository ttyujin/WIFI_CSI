import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_activity_dataset import (
    LABELS,
    SessionSpec,
    baseline_session_specs,
    build_timestamp_windows,
    discover_excluded_files,
    prepare_dataset,
    read_and_audit_session,
)


def make_frame(label, session_id, sequence, timestamp, **changes):
    row = {
        "record_type": "frame",
        "schema_version": "activity-csi-v1",
        "label": label,
        "session_id": session_id,
        "source": "esp32",
        "timestamp": timestamp,
        "server_tick": sequence,
        "node_id": 1,
        "sequence": sequence,
        "rssi_dbm": -48.0,
        "subcarrier_count": 306,
        "amplitude_count": 306,
        "amplitude": [float(sequence % 7)] * 306,
        # These inference fields are intentionally irrelevant to preparation.
        "persons": [{"id": 99}],
        "person_count": 99,
        "motion_level": "must_not_be_used",
    }
    row.update(changes)
    return row


def write_session(path, label, session_id, frames, extra_lines=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "record_type": "session",
        "schema_version": "activity-csi-v1",
        "label": label,
        "session_id": session_id,
        "source": "esp32",
        "timestamp": 999.0,
    }
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata) + "\n")
        for frame in frames:
            stream.write(json.dumps(frame) + "\n")
        for line in extra_lines:
            stream.write(line + "\n")


def populate_baseline(root):
    for spec in baseline_session_specs(root):
        frames = [
            make_frame(spec.label, spec.session_id, index, 1_000.0 + index)
            for index in range(7)
        ]
        write_session(spec.path, spec.label, spec.session_id, frames)


class PrepareActivityDatasetTests(unittest.TestCase):
    def test_exact_24_session_allowlist_excludes_nonbaseline_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "activity"
            populate_baseline(root)
            extras = (
                root / "SITTING" / "sitting_pilot_001.jsonl",
                root / "SITTING" / "sitting_test_001.jsonl",
                root / "SITTING" / "sitting_diag_001.jsonl",
                root / "MOVING" / "moving_009.jsonl",
                root / "MOVING" / "moving_010.jsonl",
                root / "MOVING" / "moving_011.jsonl",
                root / "LYING" / "lying_009.jsonl",
                root / "LYING" / "lying_011.jsonl",
            )
            for path in extras:
                path.write_text("{}\n", encoding="utf-8")
            specs = baseline_session_specs(root)
            self.assertEqual(len(specs), 24)
            self.assertEqual(
                [spec.session_id for spec in specs],
                [f"{label.lower()}_{number:03d}" for label in LABELS for number in range(1, 9)],
            )
            excluded = discover_excluded_files(root, specs)
            self.assertEqual(len(excluded), len(extras))
            self.assertTrue(all("pilot" in value or "test" in value or "diag" in value or "009" in value or "010" in value or "011" in value for value in excluded))

    def test_timestamp_windows_use_seconds_not_frame_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "SITTING" / "sitting_001.jsonl"
            timestamps = [0.0, 0.1, 1.0, 2.2, 3.1, 4.49, 4.51, 6.0]
            frames = [make_frame("SITTING", "sitting_001", i, ts) for i, ts in enumerate(timestamps)]
            write_session(path, "SITTING", "sitting_001", frames)
            audit = read_and_audit_session(SessionSpec("SITTING", "sitting_001", path, "SITTING/sitting_001.jsonl"))
            windows = build_timestamp_windows(audit, 3.0, 1.5)
            self.assertEqual([(w.start_timestamp, w.end_timestamp) for w in windows], [(0.0, 3.0), (1.5, 4.5), (3.0, 6.0)])
            self.assertEqual([w.frame_count for w in windows], [4, 3, 3])

    def test_windows_never_cross_session_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audits = []
            for label, session_id in (("SITTING", "sitting_001"), ("MOVING", "moving_001")):
                path = root / label / f"{session_id}.jsonl"
                frames = [make_frame(label, session_id, i, float(i)) for i in range(7)]
                write_session(path, label, session_id, frames)
                audits.append(read_and_audit_session(SessionSpec(label, session_id, path, f"{label}/{session_id}.jsonl")))
            windows = []
            for audit in audits:
                session_windows = build_timestamp_windows(audit)
                self.assertTrue(
                    all(window.session_id == audit.spec.session_id for window in session_windows)
                )
                windows.extend(session_windows)
            self.assertEqual({window.session_id for window in windows}, {"sitting_001", "moving_001"})
            self.assertTrue(all(window.source_file.lower().find(window.session_id) >= 0 for window in windows))

    def test_306_amplitude_validation_and_nonfinite_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "SITTING" / "sitting_001.jsonl"
            frames = [
                make_frame("SITTING", "sitting_001", 1, 1.0),
                make_frame("SITTING", "sitting_001", 2, 2.0, amplitude=[1.0] * 305),
                make_frame("SITTING", "sitting_001", 3, 3.0, amplitude=[None] * 306),
            ]
            write_session(path, "SITTING", "sitting_001", frames)
            audit = read_and_audit_session(SessionSpec("SITTING", "sitting_001", path, "SITTING/sitting_001.jsonl"))
            self.assertEqual(audit.valid_frames, 1)
            self.assertEqual(audit.excluded_frames, 2)
            self.assertEqual(audit.error_reasons["amplitude_length_not_306"], 1)
            self.assertEqual(audit.error_reasons["invalid_amplitude_value"], 1)

    def test_malformed_frame_and_json_are_excluded_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "LYING" / "lying_001.jsonl"
            frames = [
                make_frame("LYING", "lying_001", 1, 1.0),
                make_frame("LYING", "lying_001", 2, 2.0, source="simulator"),
            ]
            write_session(path, "LYING", "lying_001", frames, extra_lines=("{broken",))
            audit = read_and_audit_session(SessionSpec("LYING", "lying_001", path, "LYING/lying_001.jsonl"))
            self.assertEqual(audit.status, "WARN")
            self.assertEqual(audit.valid_frames, 1)
            self.assertEqual(audit.excluded_frames, 1)
            self.assertEqual(audit.parse_errors, 1)
            self.assertEqual(audit.error_reasons["source_not_esp32"], 1)

    def test_output_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "activity"
            output_one = Path(temp) / "one"
            output_two = Path(temp) / "two"
            populate_baseline(root)
            first = prepare_dataset(root, output_one)
            second = prepare_dataset(root, output_two)
            self.assertEqual(len(first.windows), len(second.windows))
            for filename in ("windows_manifest.csv", "session_summary.csv", "amplitude_audit.csv"):
                self.assertEqual((output_one / filename).read_bytes(), (output_two / filename).read_bytes())


if __name__ == "__main__":
    unittest.main()
