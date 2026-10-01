#!/usr/bin/env python3
"""Train and validate the reproducible offline activity-v1 artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

try:
    from tools.activity_inference import (
        ActivityV1Predictor,
        FEATURE_DIMENSION,
        MODEL_ID,
        build_inference_feature,
    )
    from tools.train_activity_baseline import (
        EXPECTED_SUBCARRIERS,
        LABELS,
        RANDOM_STATE,
        RAW_FEATURE_NAMES,
        WindowSpec,
        _load_session_frames,
        build_window_features,
        fit_model,
        load_baseline_windows,
    )
    from tools.prepare_activity_dataset import SessionSpec, baseline_session_specs
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from activity_inference import (  # type: ignore[no-redef]
        ActivityV1Predictor,
        FEATURE_DIMENSION,
        MODEL_ID,
        build_inference_feature,
    )
    from train_activity_baseline import (  # type: ignore[no-redef]
        EXPECTED_SUBCARRIERS,
        LABELS,
        RANDOM_STATE,
        RAW_FEATURE_NAMES,
        WindowSpec,
        _load_session_frames,
        build_window_features,
        fit_model,
        load_baseline_windows,
    )
    from prepare_activity_dataset import SessionSpec, baseline_session_specs  # type: ignore[no-redef]


WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.5
ROUND_TRIP_ATOL = 1e-12
FEATURE_EQUIVALENCE_ATOL = 1e-12
BASELINE_FEATURE_SNAPSHOT_ATOL = 1e-8
DIAGNOSTIC_SESSIONS = (("MOVING", "moving_011"), ("LYING", "lying_011"))


def training_session_specs(activity_root: str | Path) -> list[SessionSpec]:
    return baseline_session_specs(activity_root)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_data_fingerprint(specs: Sequence[SessionSpec], manifest: Path) -> str:
    digest = hashlib.sha256()
    for spec in specs:
        digest.update(spec.source_file.encode("utf-8"))
        digest.update(_sha256_file(spec.path).encode("ascii"))
    digest.update(manifest.as_posix().encode("utf-8"))
    digest.update(_sha256_file(manifest).encode("ascii"))
    return digest.hexdigest()


def build_feature_spec() -> dict[str, Any]:
    return {
        "schema_version": "activity-feature-spec-v1",
        "feature_type": "frame_mean_normalized_subcarrier_temporal_statistics",
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "window_interval": "half_open_[start,end)",
        "expected_subcarriers": EXPECTED_SUBCARRIERS,
        "frame_normalization": "amplitude / frame_amplitude_mean",
        "zero_division_safety": {
            "epsilon": 1e-12,
            "condition": "abs(frame_amplitude_mean) <= epsilon",
            "behavior": "replace_entire_normalized_frame_with_zeros",
        },
        "feature_blocks_in_order": [
            {"name": "subcarrier_temporal_mean", "length": 306, "indices": [0, 305]},
            {"name": "subcarrier_temporal_population_std", "length": 306, "indices": [306, 611]},
            {"name": "subcarrier_mean_absolute_temporal_difference", "length": 306, "indices": [612, 917]},
        ],
        "temporal_difference_single_frame_behavior": "306 zeros",
        "feature_dimension": FEATURE_DIMENSION,
        "feature_names": list(RAW_FEATURE_NAMES),
        "classes": list(LABELS),
        "rssi_included": False,
        "excluded_inputs": [
            "persons", "keypoints", "person_count", "motion_level",
            "existing_ruview_classification",
        ],
    }


def build_metadata(created_at: str, fingerprint: str, model_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "activity-model-metadata-v1",
        "model_id": MODEL_ID,
        "model_type": "multinomial_logistic_regression_with_standard_scaler",
        "classes": list(LABELS),
        "feature_type": "frame_mean_normalized_subcarrier_temporal_statistics",
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "expected_subcarriers": EXPECTED_SUBCARRIERS,
        "feature_dimension": FEATURE_DIMENSION,
        "created_at": created_at,
        "baseline_session_count": 24,
        "random_state": RANDOM_STATE,
        "training_data_fingerprint_sha256": fingerprint,
        "model_sha256": model_sha256,
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def _resolve_created_at(output_dir: Path, fingerprint: str, override: str | None) -> str:
    if override:
        return override
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if previous.get("training_data_fingerprint_sha256") == fingerprint:
                return str(previous["created_at"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_normalized_logistic_cv_summary(path: Path) -> dict[str, Any]:
    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("feature_set") != "NORMALIZED" or row.get("model") != "LOGISTIC_REGRESSION":
                continue
            selected[row["level"]] = {
                key: (float(value) if key.endswith(("_mean", "_std")) else value)
                for key, value in row.items()
                if key in {
                    "level", "accuracy_mean", "accuracy_std", "macro_precision_mean",
                    "macro_precision_std", "macro_recall_mean", "macro_recall_std",
                    "macro_f1_mean", "macro_f1_std",
                }
            }
    if set(selected) != {"window", "session"}:
        raise ValueError("validated NORMALIZED Logistic CV summary is missing")
    return selected


def validate_baseline_feature_snapshot(
    features: Sequence[Any], path: Path
) -> float:
    expected = {feature.window.window_id: feature.normalized for feature in features}
    seen: set[str] = set()
    maximum_difference = 0.0
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            window_id = row.get("window_id", "")
            if window_id not in expected:
                raise ValueError(f"unexpected window in baseline feature snapshot: {window_id}")
            if window_id in seen:
                raise ValueError(f"duplicate window in baseline feature snapshot: {window_id}")
            seen.add(window_id)
            saved = np.asarray([float(row[name]) for name in RAW_FEATURE_NAMES])
            difference = float(np.max(np.abs(saved - expected[window_id])))
            maximum_difference = max(maximum_difference, difference)
    if seen != set(expected):
        raise ValueError("baseline feature snapshot does not contain the same windows")
    if maximum_difference > BASELINE_FEATURE_SNAPSHOT_ATOL:
        raise ValueError(
            f"baseline feature snapshot differs by {maximum_difference:.3e} "
            f"> {BASELINE_FEATURE_SNAPSHOT_ATOL:.3e}"
        )
    return maximum_difference


def _amplitude_for_window(spec: SessionSpec, window: WindowSpec) -> np.ndarray:
    frames = _load_session_frames(spec)
    selected = [
        frame for frame in frames if window.start_timestamp <= frame.timestamp < window.end_timestamp
    ]
    if len(selected) != window.frame_count:
        raise ValueError(f"round-trip window frame count mismatch: {window.window_id}")
    return np.stack([frame.amplitude for frame in selected])


def _diagnostic_windows(spec: SessionSpec) -> tuple[list[WindowSpec], list[Any]]:
    frames = _load_session_frames(spec)
    first_timestamp = frames[0].timestamp
    last_timestamp = frames[-1].timestamp
    windows: list[WindowSpec] = []
    amplitudes: list[Any] = []
    index = 0
    while True:
        start = first_timestamp + index * HOP_SECONDS
        end = start + WINDOW_SECONDS
        if end > last_timestamp + 1e-9:
            break
        selected = [frame for frame in frames if start <= frame.timestamp < end]
        if selected:
            windows.append(WindowSpec(
                spec.label, spec.session_id, spec.session_id, index,
                start, end, WINDOW_SECONDS, len(selected),
            ))
            amplitudes.append(np.stack([frame.amplitude for frame in selected]))
        index += 1
    return windows, amplitudes


def run_held_out_diagnostics(
    predictor: ActivityV1Predictor, activity_root: Path
) -> dict[str, Any]:
    rows = []
    for label, session_id in DIAGNOSTIC_SESSIONS:
        path = activity_root / label / f"{session_id}.jsonl"
        spec = SessionSpec(label, session_id, path, f"{label}/{session_id}.jsonl")
        windows, amplitudes = _diagnostic_windows(spec)
        predictions = [predictor.predict(amplitude) for amplitude in amplitudes]
        counts = Counter(prediction["activity"] for prediction in predictions)
        mean_probabilities = {
            class_label: sum(
                prediction["probabilities"][class_label] for prediction in predictions
            ) / len(predictions)
            for class_label in LABELS
        }
        rows.append({
            "session_id": session_id,
            "reference_label_for_description_only": label,
            "window_count": len(windows),
            "predicted_class_counts": {class_label: counts[class_label] for class_label in LABELS},
            "predicted_class_ratios": {
                class_label: counts[class_label] / len(predictions) for class_label in LABELS
            },
            "mean_probabilities": mean_probabilities,
            "session_probability_average_prediction": max(
                LABELS, key=lambda class_label: mean_probabilities[class_label]
            ),
            "used_for_training": False,
            "used_for_model_selection_or_tuning": False,
            "formal_external_accuracy_calculated": False,
            "interpretation": "post-reboot/different-RSSI diagnostic observation only",
        })
    return {
        "schema_version": "activity-held-out-diagnostic-v1",
        "model_id": MODEL_ID,
        "sessions": rows,
        "limitation": "SITTING_011 is unavailable; this is not a 3-class external accuracy estimate",
    }


def build_activity_v1_artifact(
    activity_root: str | Path,
    windows_manifest: str | Path,
    baseline_summary: str | Path,
    baseline_feature_snapshot: str | Path,
    output_dir: str | Path,
    created_at: str | None = None,
) -> dict[str, Any]:
    activity_root = Path(activity_root)
    manifest_path = Path(windows_manifest)
    output_dir = Path(output_dir)
    specs = training_session_specs(activity_root)
    if len(specs) != 24 or any(
        token in spec.session_id for spec in specs for token in ("pilot", "test", "diag")
    ):
        raise RuntimeError("final training allowlist must contain exactly 24 formal sessions")
    windows, excluded = load_baseline_windows(manifest_path, specs)
    if excluded:
        raise RuntimeError(f"unexpected excluded sessions in validated manifest: {excluded}")
    features = build_window_features(specs, windows)
    baseline_snapshot_difference = validate_baseline_feature_snapshot(
        features, Path(baseline_feature_snapshot)
    )
    x_train = np.stack([feature.normalized for feature in features])
    y_train = np.asarray([feature.window.label for feature in features], dtype=object)
    if x_train.shape != (len(features), FEATURE_DIMENSION):
        raise RuntimeError("final training feature dimension mismatch")
    model = fit_model("LOGISTIC_REGRESSION", x_train, y_train)

    validation_amplitude = _amplitude_for_window(specs[0], windows[0])
    inference_feature, zero_mean_count = build_inference_feature(validation_amplitude)
    training_feature = features[0].normalized
    feature_max_abs_difference = float(np.max(np.abs(training_feature - inference_feature)))
    if not np.allclose(
        training_feature, inference_feature, rtol=0.0, atol=FEATURE_EQUIVALENCE_ATOL
    ):
        raise RuntimeError("training and inference feature extraction differ")

    before_save = model.predict_proba(x_train[:8])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    temporary_model = output_dir / "model.joblib.tmp"
    joblib.dump(model, temporary_model, compress=0)
    temporary_model.replace(model_path)
    loaded_model = joblib.load(model_path)
    after_load = loaded_model.predict_proba(x_train[:8])
    prediction_max_abs_difference = float(np.max(np.abs(before_save - after_load)))
    if not np.allclose(before_save, after_load, rtol=0.0, atol=ROUND_TRIP_ATOL):
        raise RuntimeError("model prediction changed after joblib round trip")
    if set(str(value) for value in loaded_model.classes_) != set(LABELS):
        raise RuntimeError("saved model class set mismatch")

    fingerprint = training_data_fingerprint(specs, manifest_path)
    artifact_created_at = _resolve_created_at(output_dir, fingerprint, created_at)
    feature_spec = build_feature_spec()
    model_sha256 = _sha256_file(model_path)
    metadata = build_metadata(artifact_created_at, fingerprint, model_sha256)
    _atomic_json(output_dir / "metadata.json", metadata)
    _atomic_json(output_dir / "feature_spec.json", feature_spec)

    predictor = ActivityV1Predictor.from_artifact_dir(output_dir)
    helper_prediction = predictor.predict(validation_amplitude)
    diagnostics = run_held_out_diagnostics(predictor, activity_root)
    _atomic_json(output_dir / "held_out_diagnostics.json", diagnostics)
    cv_summary = _load_normalized_logistic_cv_summary(Path(baseline_summary))
    training_summary = {
        "schema_version": "activity-training-summary-v1",
        "model_id": MODEL_ID,
        "training_sessions": [spec.session_id for spec in specs],
        "training_session_count": len(specs),
        "training_window_count": len(features),
        "feature_dimension": FEATURE_DIMENSION,
        "classes": list(LABELS),
        "random_state": RANDOM_STATE,
        "model_configuration": {
            "pipeline": ["StandardScaler", "LogisticRegression"],
            "logistic_solver": "lbfgs",
            "logistic_C": 1.0,
            "logistic_max_iter": 3000,
            "multiclass_behavior": "multinomial for three classes",
            "scaler_fit_scope": "all 24 baseline sessions for final artifact",
        },
        "validated_grouped_cv": {
            "source": str(Path(baseline_summary).as_posix()),
            "split": "StratifiedGroupKFold(n_splits=4, group=session_id, random_state=42)",
            "window": cv_summary["window"],
            "session": cv_summary["session"],
        },
        "full_training_fit_accuracy_calculated": False,
        "full_training_fit_accuracy_note": "not used as performance evidence",
        "round_trip_validation": {
            "sample_windows": 8,
            "absolute_tolerance": ROUND_TRIP_ATOL,
            "relative_tolerance": 0.0,
            "max_absolute_probability_difference": prediction_max_abs_difference,
            "passed": True,
        },
        "training_inference_feature_equivalence": {
            "window_id": windows[0].window_id,
            "absolute_tolerance": FEATURE_EQUIVALENCE_ATOL,
            "relative_tolerance": 0.0,
            "max_absolute_feature_difference": feature_max_abs_difference,
            "zero_mean_frame_count": zero_mean_count,
            "passed": True,
        },
        "validated_baseline_feature_snapshot": {
            "source": str(Path(baseline_feature_snapshot).as_posix()),
            "absolute_tolerance": BASELINE_FEATURE_SNAPSHOT_ATOL,
            "max_absolute_feature_difference": baseline_snapshot_difference,
            "passed": True,
        },
        "inference_smoke_prediction": helper_prediction,
        "training_data_fingerprint_sha256": fingerprint,
        "scope_limitation": (
            "within-subject / within-environment session-level baseline; "
            "not elderly or general-environment performance"
        ),
        "held_out_diagnostics_file": "held_out_diagnostics.json",
        "excluded_from_training": ["all pilot/test/diag sessions", "moving_009-011", "lying_009/011"],
    }
    _atomic_json(output_dir / "training_summary.json", training_summary)
    return {
        "metadata": metadata,
        "feature_spec": feature_spec,
        "training_summary": training_summary,
        "diagnostics": diagnostics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build reproducible offline activity-v1 artifact")
    parser.add_argument("activity_root", type=Path)
    parser.add_argument(
        "--windows-manifest", type=Path,
        default=Path("data/processed/activity_v1/windows_manifest.csv"),
    )
    parser.add_argument(
        "--baseline-summary", type=Path,
        default=Path("data/processed/activity_v1/baseline/baseline_summary.csv"),
    )
    parser.add_argument(
        "--baseline-feature-snapshot", type=Path,
        default=Path("data/processed/activity_v1/baseline/window_features_normalized.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/activity_v1"))
    parser.add_argument("--created-at", help="Optional fixed ISO-8601 value for reproducible builds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_activity_v1_artifact(
        args.activity_root,
        args.windows_manifest,
        args.baseline_summary,
        args.baseline_feature_snapshot,
        args.output_dir,
        args.created_at,
    )
    print(f"Built {MODEL_ID} at {args.output_dir}")
    print("Training sessions: 24; feature: NORMALIZED; dimension: 918")
    round_trip = result["training_summary"]["round_trip_validation"]
    equivalence = result["training_summary"]["training_inference_feature_equivalence"]
    print(
        f"Save/load max probability difference: "
        f"{round_trip['max_absolute_probability_difference']:.3e} "
        f"(atol={round_trip['absolute_tolerance']:.1e})"
    )
    print(
        f"Training/inference max feature difference: "
        f"{equivalence['max_absolute_feature_difference']:.3e} "
        f"(atol={equivalence['absolute_tolerance']:.1e})"
    )
    for diagnostic in result["diagnostics"]["sessions"]:
        print(
            f"{diagnostic['session_id']}: windows={diagnostic['window_count']} "
            f"counts={diagnostic['predicted_class_counts']} "
            f"session_prediction={diagnostic['session_probability_average_prediction']}"
        )
    print("Held-out 011 results are diagnostic only; no 3-class external accuracy calculated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
