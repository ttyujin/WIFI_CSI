#!/usr/bin/env python3
"""Reusable offline inference helper for the activity-v1 model artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    from tools.train_activity_baseline import (
        EXPECTED_SUBCARRIERS,
        LABELS,
        RAW_FEATURE_NAMES,
        extract_normalized_features,
    )
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from train_activity_baseline import (  # type: ignore[no-redef]
        EXPECTED_SUBCARRIERS,
        LABELS,
        RAW_FEATURE_NAMES,
        extract_normalized_features,
    )


MODEL_ID = "activity-v1"
FEATURE_DIMENSION = len(RAW_FEATURE_NAMES)


def amplitude_matrix_from_window(window: Any) -> np.ndarray:
    """Convert an ndarray or frame dictionaries into a finite frame matrix."""

    if isinstance(window, np.ndarray):
        matrix = np.asarray(window, dtype=np.float64)
    else:
        frames = window.get("frames") if isinstance(window, Mapping) else window
        if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
            raise ValueError("window must be a 2-D amplitude array or a sequence of frames")
        amplitudes: list[Any] = []
        for index, frame in enumerate(frames):
            if isinstance(frame, Mapping):
                if "amplitude" not in frame:
                    raise ValueError(f"frame {index} is missing amplitude")
                amplitudes.append(frame["amplitude"])
            else:
                amplitudes.append(frame)
        try:
            matrix = np.asarray(amplitudes, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("amplitude contains a non-numeric value") from error
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != EXPECTED_SUBCARRIERS:
        raise ValueError(
            f"window amplitude must have shape (frames, {EXPECTED_SUBCARRIERS})"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("window amplitude contains NaN or infinity")
    return matrix


def build_inference_feature(window: Any) -> tuple[np.ndarray, int]:
    """Apply the exact validated baseline NORMALIZED feature extraction."""

    matrix = amplitude_matrix_from_window(window)
    feature, zero_mean_frame_count = extract_normalized_features(matrix)
    if feature.shape != (FEATURE_DIMENSION,) or not np.all(np.isfinite(feature)):
        raise ValueError("inference feature has an invalid shape or non-finite value")
    return feature, zero_mean_frame_count


class ActivityV1Predictor:
    def __init__(self, model: Any, metadata: dict[str, Any], feature_spec: dict[str, Any]):
        self.model = model
        self.metadata = metadata
        self.feature_spec = feature_spec
        self._validate_artifact()

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str | Path) -> "ActivityV1Predictor":
        root = Path(artifact_dir)
        with (root / "metadata.json").open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        with (root / "feature_spec.json").open("r", encoding="utf-8") as stream:
            feature_spec = json.load(stream)
        model = joblib.load(root / "model.joblib")
        return cls(model, metadata, feature_spec)

    def _validate_artifact(self) -> None:
        if self.metadata.get("model_id") != MODEL_ID:
            raise ValueError("unexpected model_id")
        if tuple(self.metadata.get("classes", ())) != LABELS:
            raise ValueError("artifact class order does not match activity-v1")
        if self.metadata.get("feature_dimension") != FEATURE_DIMENSION:
            raise ValueError("artifact feature dimension does not match inference code")
        if self.feature_spec.get("feature_names") != list(RAW_FEATURE_NAMES):
            raise ValueError("artifact feature ordering does not match inference code")
        if self.feature_spec.get("expected_subcarriers") != EXPECTED_SUBCARRIERS:
            raise ValueError("artifact subcarrier count does not match inference code")
        model_classes = set(str(value) for value in self.model.classes_)
        if model_classes != set(LABELS):
            raise ValueError("model classes do not match activity-v1 classes")

    def predict(self, window: Any) -> dict[str, Any]:
        feature, zero_mean_frame_count = build_inference_feature(window)
        raw_probabilities = self.model.predict_proba(feature.reshape(1, -1))[0]
        model_classes = [str(value) for value in self.model.classes_]
        probabilities = {
            label: float(raw_probabilities[model_classes.index(label)]) for label in LABELS
        }
        activity = max(LABELS, key=lambda label: probabilities[label])
        return {
            "activity": activity,
            "confidence": probabilities[activity],
            "probabilities": probabilities,
            "model_id": MODEL_ID,
            "zero_mean_frame_count": zero_mean_frame_count,
        }


def _read_window(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        frames = []
        for line in text.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and row.get("record_type") == "frame":
                frames.append(row)
        value = frames
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline activity-v1 window inference")
    parser.add_argument("window_file", type=Path, help="JSON array/object or JSONL frame window")
    parser.add_argument("--model-dir", type=Path, default=Path("models/activity_v1"))
    args = parser.parse_args()
    predictor = ActivityV1Predictor.from_artifact_dir(args.model_dir)
    print(json.dumps(predictor.predict(_read_window(args.window_file)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
