import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.activity_qc import (
    MANIFEST_COLUMNS,
    QcHeuristics,
    analyze_recording,
    dataset_role,
    scan_recordings,
    write_manifest,
)


def frame(sequence: int, timestamp: float, **changes):
    value = {
        "record_type": "frame",
        "schema_version": "activity-csi-v1",
        "session_id": "sitting_001",
        "label": "SITTING",
        "timestamp": timestamp,
        "server_tick": sequence,
        "node_id": 1,
        "sequence": sequence,
        "source": "esp32",
        "rssi_dbm": -48.0,
        "subcarrier_count": 306,
        "amplitude_count": 306,
        "amplitude": [1.0] * 306,
        "features": {"variance": 0.1},
    }
    value.update(changes)
    return value


def write_recording(path: Path, frames, *, session_id="sitting_001", label="SITTING"):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "record_type": "session",
        "schema_version": "activity-csi-v1",
        "session_id": session_id,
        "label": label,
        "source": "esp32",
        "timestamp": 1_000.0,
    }
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata) + "\n")
        for item in frames:
            item = dict(item)
            item["session_id"] = session_id
            item["label"] = label
            stream.write(json.dumps(item) + "\n")


RELAXED = QcHeuristics(
    fail_min_duration_sec=0.0,
    duration_warn_min_sec=0.0,
    duration_warn_max_sec=120.0,
    fps_warn_min=0.0,
    fps_warn_max=1_000.0,
)


class ActivityQcTests(unittest.TestCase):
    def test_valid_306_amplitude_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "SITTING" / "sitting_001.jsonl"
            write_recording(path, [frame(10, 1_000.0), frame(11, 1_000.033)])
            result = analyze_recording(path, RELAXED)
            self.assertEqual(result.qc_status, "PASS")
            self.assertEqual(result.subcarrier_counts, (306,))
            self.assertEqual(result.amplitude_counts, (306,))
            self.assertEqual(result.amplitude_lengths, (306,))
            self.assertEqual(result.invalid_amplitude_frames, 0)

    def test_wrong_amplitude_length_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "SITTING" / "sitting_001.jsonl"
            write_recording(path, [frame(10, 1_000.0, amplitude=[1.0] * 305)])
            result = analyze_recording(path, RELAXED)
            self.assertEqual(result.qc_status, "FAIL")
            self.assertTrue(any(reason.startswith("amplitude_length_not_306") for reason in result.fail_reasons))

    def test_empty_file_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "SITTING" / "empty.jsonl"
            path.parent.mkdir(parents=True)
            path.touch()
            result = analyze_recording(path)
            self.assertEqual(result.qc_status, "FAIL")
            self.assertIn("frame_count_zero", result.fail_reasons)

    def test_invalid_json_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "SITTING" / "broken.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("{not json}\n", encoding="utf-8")
            result = analyze_recording(path)
            self.assertEqual(result.qc_status, "FAIL")
            self.assertEqual(result.parse_error_count, 1)

    def test_timestamp_reversal_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "SITTING" / "sitting_001.jsonl"
            write_recording(path, [frame(10, 1_001.0), frame(11, 1_000.0)])
            result = analyze_recording(path, RELAXED)
            self.assertEqual(result.timestamp_inversions, 1)
            self.assertFalse(result.timestamps_monotonic)
            self.assertEqual(result.qc_status, "FAIL")

    def test_sequence_gap_calculation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "SITTING" / "sitting_001.jsonl"
            write_recording(path, [frame(10, 1_000.0), frame(13, 1_000.1)])
            result = analyze_recording(path, RELAXED)
            self.assertEqual(result.sequence_gap_count, 2)
            self.assertAlmostEqual(result.sequence_gap_rate, 2 / 4)

    def test_sequence_gap_warning_bands_follow_observed_collection_range(self):
        cases = (
            (4, "PASS", None),
            (6, "WARN", "sequence_gap_rate_warning"),
            (11, "WARN", "sequence_gap_rate_strong_warning"),
        )
        with tempfile.TemporaryDirectory() as temp:
            for missing_count, expected_status, expected_reason in cases:
                with self.subTest(missing_count=missing_count):
                    path = Path(temp) / "SITTING" / f"gap_{missing_count}.jsonl"
                    missing = set(range(20, 20 + missing_count))
                    sequences = [value for value in range(1, 101) if value not in missing]
                    frames = [
                        frame(sequence, 1_000.0 + index * 0.033)
                        for index, sequence in enumerate(sequences)
                    ]
                    write_recording(path, frames)
                    result = analyze_recording(path, RELAXED)
                    self.assertEqual(result.sequence_gap_count, missing_count)
                    self.assertAlmostEqual(result.sequence_gap_rate, missing_count / 100)
                    self.assertEqual(result.qc_status, expected_status)
                    gap_reasons = [
                        reason
                        for reason in result.warn_reasons
                        if reason.startswith("sequence_gap_rate_")
                    ]
                    if expected_reason is None:
                        self.assertEqual(gap_reasons, [])
                    else:
                        self.assertTrue(
                            any(reason.startswith(expected_reason) for reason in gap_reasons)
                        )

    def test_duplicate_sequence_count(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "SITTING" / "sitting_001.jsonl"
            write_recording(
                path,
                [frame(10, 1_000.0), frame(10, 1_000.01), frame(11, 1_000.02)],
            )
            result = analyze_recording(path, RELAXED)
            self.assertEqual(result.duplicate_sequences, 1)
            self.assertEqual(result.sequence_gap_count, 0)
            self.assertEqual(result.qc_status, "WARN")

    def test_scan_multiple_recording_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "activity"
            write_recording(root / "SITTING" / "sitting_001.jsonl", [frame(1, 1.0)])
            write_recording(
                root / "MOVING" / "moving_001.jsonl",
                [frame(2, 1.0, label="MOVING")],
                session_id="moving_001",
                label="MOVING",
            )
            write_recording(
                root / "LYING" / "lying_001.jsonl",
                [frame(3, 1.0, label="LYING")],
                session_id="lying_001",
                label="LYING",
            )
            results = scan_recordings(root)
            self.assertEqual(len(results), 3)
            self.assertEqual(
                {result.label for result in results},
                {"SITTING", "MOVING", "LYING"},
            )

    def test_pilot_and_formal_roles(self):
        self.assertEqual(dataset_role("sitting_pilot_001"), "pilot")
        self.assertEqual(dataset_role("sitting_test_001"), "pilot")
        self.assertEqual(dataset_role("lying_pilot_003_retry"), "pilot")
        self.assertEqual(dataset_role("sitting_001"), "formal")
        self.assertEqual(dataset_role("moving_001"), "formal")

    def test_manifest_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "activity"
            source = root / "SITTING" / "sitting_pilot_001.jsonl"
            write_recording(source, [frame(1, 1.0)], session_id="sitting_pilot_001")
            results = scan_recordings(root)
            before = source.read_bytes()
            manifest = write_manifest(results, root / "dataset_manifest.csv")
            self.assertEqual(source.read_bytes(), before, "QC must not modify source JSONL")
            with manifest.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(tuple(rows[0].keys()), MANIFEST_COLUMNS)
            self.assertEqual(rows[0]["session_id"], "sitting_pilot_001")
            self.assertEqual(rows[0]["dataset_role"], "pilot")
            self.assertEqual(rows[0]["subcarrier_count"], "306")
            self.assertIn(rows[0]["qc_status"], {"PASS", "WARN", "FAIL"})


if __name__ == "__main__":
    unittest.main()
