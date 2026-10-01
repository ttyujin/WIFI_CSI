import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from tools.activity_inference import (
    FEATURE_DIMENSION,
    ActivityV1Predictor,
    amplitude_matrix_from_window,
    build_inference_feature,
)
from tools.build_activity_v1_model import (
    build_feature_spec,
    build_metadata,
    training_session_specs,
)
from tools.train_activity_baseline import LABELS, extract_feature_vectors, fit_model


class ActivityV1ArtifactTests(unittest.TestCase):
    def test_final_training_allowlist_is_exactly_24_sessions(self):
        specs = training_session_specs("activity")
        self.assertEqual(len(specs), 24)
        self.assertEqual(
            [spec.session_id for spec in specs],
            [f"{label}_{index:03d}" for label in ("sitting", "moving", "lying") for index in range(1, 9)],
        )
        self.assertFalse(any(
            token in spec.session_id
            for spec in specs
            for token in ("pilot", "test", "diag", "009", "010", "011")
        ))

    def test_feature_spec_dimension_and_class_order(self):
        feature_spec = build_feature_spec()
        metadata = build_metadata("2026-01-01T00:00:00Z", "fingerprint", "modelhash")
        self.assertEqual(FEATURE_DIMENSION, 918)
        self.assertEqual(feature_spec["feature_dimension"], 918)
        self.assertEqual(len(feature_spec["feature_names"]), 918)
        self.assertEqual(feature_spec["classes"], ["SITTING", "MOVING", "LYING"])
        self.assertEqual(metadata["classes"], ["SITTING", "MOVING", "LYING"])

    def test_training_and_inference_features_are_numerically_identical(self):
        generator = np.random.default_rng(42)
        amplitude = generator.uniform(0.5, 30.0, size=(91, 306))
        rssi = generator.uniform(-55.0, -40.0, size=91)
        _, training_normalized, _, _ = extract_feature_vectors(amplitude, rssi)
        inference_normalized, _ = build_inference_feature(amplitude)
        np.testing.assert_allclose(training_normalized, inference_normalized, rtol=0.0, atol=1e-12)

    def test_malformed_and_zero_mean_frames_are_safe(self):
        with self.assertRaises(ValueError):
            amplitude_matrix_from_window([{"amplitude": [1.0] * 305}])
        with self.assertRaises(ValueError):
            amplitude_matrix_from_window([{"amplitude": [float("nan")] * 306}])
        feature, zero_count = build_inference_feature(np.zeros((3, 306)))
        self.assertEqual(zero_count, 3)
        self.assertTrue(np.all(np.isfinite(feature)))
        self.assertTrue(np.all(feature == 0.0))

    def test_model_save_load_prediction_and_class_order(self):
        generator = np.random.default_rng(7)
        x = generator.normal(size=(30, FEATURE_DIMENSION))
        y = np.asarray(list(LABELS) * 10, dtype=object)
        model = fit_model("LOGISTIC_REGRESSION", x, y)
        before = model.predict_proba(x[:4])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.joblib"
            joblib.dump(model, path)
            loaded = joblib.load(path)
        after = loaded.predict_proba(x[:4])
        np.testing.assert_allclose(before, after, rtol=0.0, atol=1e-12)
        self.assertEqual(set(loaded.classes_), set(LABELS))

    def test_predictor_preserves_metadata_class_order_and_real_probabilities(self):
        generator = np.random.default_rng(9)
        x = generator.normal(size=(30, FEATURE_DIMENSION))
        y = np.asarray(list(LABELS) * 10, dtype=object)
        model = fit_model("LOGISTIC_REGRESSION", x, y)
        metadata = build_metadata("2026-01-01T00:00:00Z", "fingerprint", "modelhash")
        predictor = ActivityV1Predictor(model, metadata, build_feature_spec())
        window = generator.uniform(1.0, 10.0, size=(90, 306))
        feature, _ = build_inference_feature(window)
        raw = model.predict_proba(feature.reshape(1, -1))[0]
        output = predictor.predict(window)
        self.assertEqual(list(output["probabilities"]), list(LABELS))
        for label in LABELS:
            self.assertAlmostEqual(
                output["probabilities"][label], raw[list(model.classes_).index(label)]
            )
        self.assertAlmostEqual(output["confidence"], max(output["probabilities"].values()))

    def test_training_is_deterministic(self):
        generator = np.random.default_rng(11)
        x = generator.normal(size=(36, 20))
        y = np.asarray(list(LABELS) * 12, dtype=object)
        first = fit_model("LOGISTIC_REGRESSION", x, y)
        second = fit_model("LOGISTIC_REGRESSION", x, y)
        np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))

    def test_json_metadata_output_can_be_deterministic(self):
        first = build_metadata("2026-01-01T00:00:00Z", "fingerprint", "modelhash")
        second = build_metadata("2026-01-01T00:00:00Z", "fingerprint", "modelhash")
        self.assertEqual(
            json.dumps(first, sort_keys=True, separators=(",", ":")),
            json.dumps(second, sort_keys=True, separators=(",", ":")),
        )


if __name__ == "__main__":
    unittest.main()
