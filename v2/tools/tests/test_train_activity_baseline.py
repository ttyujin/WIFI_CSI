import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.prepare_activity_dataset import baseline_session_specs
from tools.train_activity_baseline import (
    LABELS,
    NUISANCE_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    WindowSpec,
    _atomic_csv,
    build_model,
    extract_feature_vectors,
    load_baseline_windows,
    make_group_folds,
)


class TrainActivityBaselineTests(unittest.TestCase):
    def test_exact_24_baseline_sessions_and_excluded_manifest_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            specs = baseline_session_specs(root / "activity")
            manifest = root / "windows.csv"
            columns = (
                "label", "session_id", "group_id", "window_index",
                "start_timestamp", "end_timestamp", "duration", "frame_count",
            )
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for spec in specs:
                    writer.writerow({
                        "label": spec.label, "session_id": spec.session_id,
                        "group_id": spec.session_id, "window_index": 0,
                        "start_timestamp": 0, "end_timestamp": 3,
                        "duration": 3, "frame_count": 2,
                    })
                for label, session_id in (
                    ("SITTING", "sitting_pilot_001"),
                    ("SITTING", "sitting_test_001"),
                    ("SITTING", "sitting_diag_001"),
                    ("MOVING", "moving_009"),
                    ("MOVING", "moving_010"),
                    ("MOVING", "moving_011"),
                    ("LYING", "lying_009"),
                    ("LYING", "lying_011"),
                ):
                    writer.writerow({
                        "label": label, "session_id": session_id, "group_id": session_id,
                        "window_index": 0, "start_timestamp": 0, "end_timestamp": 3,
                        "duration": 3, "frame_count": 2,
                    })
            windows, excluded = load_baseline_windows(manifest, specs)
            self.assertEqual(len(specs), 24)
            self.assertEqual(len(windows), 24)
            self.assertEqual(len(excluded), 8)
            self.assertTrue(all(window.session_id in {spec.session_id for spec in specs} for window in windows))

    def test_deterministic_group_split_has_no_leakage_and_all_labels(self):
        specs = baseline_session_specs("activity")
        first = make_group_folds(specs)
        second = make_group_folds(specs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        for fold in first:
            self.assertFalse(set(fold.train_sessions) & set(fold.test_sessions))
            train_labels = {spec.label for spec in specs if spec.session_id in fold.train_sessions}
            test_labels = [spec.label for spec in specs if spec.session_id in fold.test_sessions]
            self.assertEqual(train_labels, set(LABELS))
            self.assertEqual({label: test_labels.count(label) for label in LABELS}, {label: 2 for label in LABELS})

    def test_raw_and_normalized_shapes_are_fixed_and_zero_mean_is_safe(self):
        amplitude = np.ones((4, 306), dtype=np.float64)
        amplitude[0] = 0.0
        rssi = np.asarray([-50.0, -49.0, -48.0, -47.0])
        raw, normalized, nuisance, zero_count = extract_feature_vectors(amplitude, rssi)
        self.assertEqual(raw.shape, (len(RAW_FEATURE_NAMES),))
        self.assertEqual(normalized.shape, (len(RAW_FEATURE_NAMES),))
        self.assertEqual(nuisance.shape, (len(NUISANCE_FEATURE_NAMES),))
        self.assertEqual(zero_count, 1)
        self.assertTrue(np.all(np.isfinite(normalized)))

    def test_logistic_scaler_is_fit_on_train_only(self):
        train = np.asarray([[0.0, 0.0], [2.0, 2.0], [0.1, 0.1], [1.9, 1.9]])
        labels = np.asarray(["SITTING", "MOVING", "SITTING", "MOVING"])
        model = build_model("LOGISTIC_REGRESSION")
        model.fit(train, labels)
        np.testing.assert_allclose(model.named_steps["scaler"].mean_, [1.0, 1.0])
        transformed_test = model.named_steps["scaler"].transform([[100.0, 100.0]])
        self.assertGreater(transformed_test[0, 0], 50.0)

    def test_nuisance_features_are_window_local_without_group_threshold(self):
        self.assertFalse(any("group" in name or "high" in name or "low" in name for name in NUISANCE_FEATURE_NAMES))
        amplitude = np.full((3, 306), 2.0)
        rssi = np.asarray([-50.0, -49.0, -48.0])
        _, _, first, _ = extract_feature_vectors(amplitude, rssi)
        _, _, second, _ = extract_feature_vectors(amplitude.copy(), rssi.copy())
        np.testing.assert_array_equal(first, second)

    def test_window_manifest_rejects_session_boundary_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            specs = baseline_session_specs(root / "activity")
            manifest = root / "windows.csv"
            columns = (
                "label", "session_id", "group_id", "window_index",
                "start_timestamp", "end_timestamp", "duration", "frame_count",
            )
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                for spec in specs:
                    writer.writerow({
                        "label": spec.label, "session_id": spec.session_id,
                        "group_id": "moving_001" if spec.session_id == "sitting_001" else spec.session_id,
                        "window_index": 0, "start_timestamp": 0, "end_timestamp": 3,
                        "duration": 3, "frame_count": 2,
                    })
            with self.assertRaises(ValueError):
                load_baseline_windows(manifest, specs)

    def test_csv_output_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            columns = ("session_id", "value")
            rows = ({"session_id": "sitting_001", "value": 1.25},)
            first = Path(temp) / "first.csv"
            second = Path(temp) / "second.csv"
            _atomic_csv(first, columns, rows)
            _atomic_csv(second, columns, rows)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
