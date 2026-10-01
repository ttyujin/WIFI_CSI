#!/usr/bin/env python3
"""Run the fixed 24-session activity baseline and confound ablation.

This is a within-subject, within-environment session-level experiment.  It
never trains or evaluates on pilot/test/diag/009-011 recordings, never splits
frames or windows at random, and never uses RuView persons/keypoints,
person_count, motion_level, or existing classifications.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as error:  # pragma: no cover - environment setup failure
    raise SystemExit(
        "scikit-learn is required. Install tools/activity_baseline_requirements.txt"
    ) from error

try:
    from tools.prepare_activity_dataset import (
        EXPECTED_SUBCARRIERS,
        LABELS,
        SessionSpec,
        _finite_number,
        _integer,
        _validate_frame,
        baseline_session_specs,
    )
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from prepare_activity_dataset import (  # type: ignore[no-redef]
        EXPECTED_SUBCARRIERS,
        LABELS,
        SessionSpec,
        _finite_number,
        _integer,
        _validate_frame,
        baseline_session_specs,
    )


RANDOM_STATE = 42
N_SPLITS = 4
FEATURE_SETS = ("RAW", "NORMALIZED", "NUISANCE_ONLY")
MODEL_NAMES = ("LOGISTIC_REGRESSION", "RANDOM_FOREST")
SESSION_PREDICTION_METHOD = "mean_class_probability"
RAW_FEATURE_NAMES = tuple(
    [f"sc_{index:03d}_temporal_mean" for index in range(EXPECTED_SUBCARRIERS)]
    + [f"sc_{index:03d}_temporal_std" for index in range(EXPECTED_SUBCARRIERS)]
    + [f"sc_{index:03d}_mean_abs_temporal_diff" for index in range(EXPECTED_SUBCARRIERS)]
)
NUISANCE_FEATURE_NAMES = (
    "frame_amplitude_mean_window_mean",
    "frame_amplitude_mean_window_std",
    "frame_amplitude_mean_q25",
    "frame_amplitude_mean_q50",
    "frame_amplitude_mean_q75",
    "frame_amplitude_mean_abs_temporal_diff",
    "rssi_window_mean",
    "rssi_window_std",
)
FEATURE_METADATA_COLUMNS = (
    "window_id", "label", "session_id", "group_id", "window_index",
    "start_timestamp", "end_timestamp", "duration", "frame_count",
    "zero_mean_frame_count",
)


@dataclass(frozen=True)
class ActivityFrame:
    timestamp: float
    rssi: float
    amplitude: np.ndarray


@dataclass(frozen=True)
class WindowSpec:
    label: str
    session_id: str
    group_id: str
    window_index: int
    start_timestamp: float
    end_timestamp: float
    duration: float
    frame_count: int

    @property
    def window_id(self) -> str:
        return f"{self.session_id}_w{self.window_index:03d}"


@dataclass(frozen=True)
class WindowFeatures:
    window: WindowSpec
    raw: np.ndarray
    normalized: np.ndarray
    nuisance: np.ndarray
    zero_mean_frame_count: int


@dataclass(frozen=True)
class GroupFold:
    fold: int
    train_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]


@dataclass(frozen=True)
class BaselineResult:
    folds: tuple[GroupFold, ...]
    feature_rows: tuple[WindowFeatures, ...]
    fold_metric_rows: tuple[dict[str, Any], ...]
    session_prediction_rows: tuple[dict[str, Any], ...]
    confusion_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]
    output_dir: Path


def _load_session_frames(spec: SessionSpec) -> list[ActivityFrame]:
    if not spec.path.is_file():
        raise ValueError(f"missing baseline session: {spec.path}")
    frames: list[ActivityFrame] = []
    metadata_count = 0
    previous_timestamp: float | None = None
    errors: Counter[str] = Counter()
    with spec.path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, ValueError):
                errors[f"line_{line_number}:json_parse_error"] += 1
                continue
            if not isinstance(row, dict):
                errors[f"line_{line_number}:json_object_required"] += 1
                continue
            if row.get("record_type") == "session":
                metadata_count += 1
                if row.get("label") != spec.label:
                    errors["metadata_label_mismatch"] += 1
                if row.get("session_id") != spec.session_id:
                    errors["metadata_session_id_mismatch"] += 1
                if row.get("source") != "esp32":
                    errors["metadata_source_not_esp32"] += 1
                continue
            if row.get("record_type") != "frame":
                errors[f"line_{line_number}:unknown_record_type"] += 1
                continue
            frame_errors, amplitude = _validate_frame(row, spec)
            timestamp = float(row["timestamp"]) if _finite_number(row.get("timestamp")) else None
            if timestamp is not None and previous_timestamp is not None and timestamp < previous_timestamp:
                frame_errors.append("timestamp_inversion")
            if frame_errors or amplitude is None:
                for reason in frame_errors or ["invalid_frame"]:
                    errors[f"line_{line_number}:{reason}"] += 1
                continue
            previous_timestamp = timestamp
            frames.append(ActivityFrame(
                timestamp=timestamp,
                rssi=float(row["rssi_dbm"]),
                amplitude=np.asarray(amplitude, dtype=np.float64),
            ))
    if metadata_count != 1:
        errors[f"metadata_count_{metadata_count}"] += 1
    if errors:
        preview = ", ".join(list(errors)[:8])
        raise ValueError(f"baseline session validation failed for {spec.session_id}: {preview}")
    if not frames:
        raise ValueError(f"baseline session contains no valid frames: {spec.session_id}")
    return frames


def load_baseline_windows(
    manifest_path: str | Path, specs: Sequence[SessionSpec]
) -> tuple[list[WindowSpec], list[str]]:
    allowed = {spec.session_id: spec for spec in specs}
    order = {spec.session_id: index for index, spec in enumerate(specs)}
    windows: list[WindowSpec] = []
    excluded_sessions: set[str] = set()
    seen: set[tuple[str, int]] = set()
    with Path(manifest_path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            session_id = row.get("session_id", "")
            if session_id not in allowed:
                if session_id:
                    excluded_sessions.add(session_id)
                continue
            spec = allowed[session_id]
            if row.get("label") != spec.label:
                raise ValueError(f"manifest label mismatch: {session_id}")
            if row.get("group_id") != session_id:
                raise ValueError(f"manifest group_id mismatch: {session_id}")
            index = int(row["window_index"])
            key = (session_id, index)
            if key in seen:
                raise ValueError(f"duplicate manifest window: {session_id}/{index}")
            seen.add(key)
            start = float(row["start_timestamp"])
            end = float(row["end_timestamp"])
            duration = float(row["duration"])
            if not all(math.isfinite(value) for value in (start, end, duration)):
                raise ValueError(f"non-finite window timestamp: {session_id}/{index}")
            if end <= start or duration <= 0:
                raise ValueError(f"invalid window duration: {session_id}/{index}")
            windows.append(WindowSpec(
                spec.label, session_id, session_id, index, start, end, duration,
                int(row["frame_count"]),
            ))
    missing = [spec.session_id for spec in specs if not any(w.session_id == spec.session_id for w in windows)]
    if missing:
        raise ValueError(f"manifest is missing baseline sessions: {', '.join(missing)}")
    windows.sort(key=lambda window: (order[window.session_id], window.window_index))
    return windows, sorted(excluded_sessions)


def extract_feature_vectors(
    amplitude: np.ndarray, rssi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if amplitude.ndim != 2 or amplitude.shape[1] != EXPECTED_SUBCARRIERS:
        raise ValueError(f"amplitude must have shape (frames, {EXPECTED_SUBCARRIERS})")
    if amplitude.shape[0] == 0 or rssi.shape != (amplitude.shape[0],):
        raise ValueError("window must contain matching amplitude and RSSI frames")

    raw = extract_temporal_features(amplitude)
    normalized, zero_mean_frame_count = extract_normalized_features(amplitude)

    frame_means = amplitude.mean(axis=1)
    nuisance = np.asarray((
        frame_means.mean(),
        frame_means.std(),
        np.quantile(frame_means, 0.25),
        np.quantile(frame_means, 0.50),
        np.quantile(frame_means, 0.75),
        np.abs(np.diff(frame_means)).mean() if len(frame_means) > 1 else 0.0,
        rssi.mean(),
        rssi.std(),
    ), dtype=np.float64)
    if not all(np.all(np.isfinite(values)) for values in (raw, normalized, nuisance)):
        raise ValueError("feature extraction produced non-finite values")
    return raw, normalized, nuisance, zero_mean_frame_count


def extract_temporal_features(amplitude: np.ndarray) -> np.ndarray:
    """Return mean, population std, mean absolute temporal difference blocks."""

    if amplitude.ndim != 2 or amplitude.shape[1] != EXPECTED_SUBCARRIERS:
        raise ValueError(f"amplitude must have shape (frames, {EXPECTED_SUBCARRIERS})")
    if amplitude.shape[0] == 0:
        raise ValueError("at least one frame is required")
    temporal_mean = amplitude.mean(axis=0)
    temporal_std = amplitude.std(axis=0)
    temporal_diff = (
        np.abs(np.diff(amplitude, axis=0)).mean(axis=0)
        if amplitude.shape[0] > 1 else np.zeros(EXPECTED_SUBCARRIERS)
    )
    return np.concatenate((temporal_mean, temporal_std, temporal_diff))


def frame_mean_normalize(amplitude: np.ndarray) -> tuple[np.ndarray, int]:
    """Divide each frame by its mean; near-zero-mean frames become all zeros."""

    if amplitude.ndim != 2 or amplitude.shape[1] != EXPECTED_SUBCARRIERS:
        raise ValueError(f"amplitude must have shape (frames, {EXPECTED_SUBCARRIERS})")
    if amplitude.shape[0] == 0 or not np.all(np.isfinite(amplitude)):
        raise ValueError("amplitude must contain finite values and at least one frame")
    frame_means = amplitude.mean(axis=1)
    nonzero = np.abs(frame_means) > 1e-12
    normalized_frames = np.zeros_like(amplitude, dtype=np.float64)
    np.divide(amplitude, frame_means[:, None], out=normalized_frames, where=nonzero[:, None])
    return normalized_frames, int((~nonzero).sum())


def extract_normalized_features(amplitude: np.ndarray) -> tuple[np.ndarray, int]:
    normalized_frames, zero_mean_frame_count = frame_mean_normalize(amplitude)
    return extract_temporal_features(normalized_frames), zero_mean_frame_count


def build_window_features(
    specs: Sequence[SessionSpec], windows: Sequence[WindowSpec]
) -> list[WindowFeatures]:
    frames_by_session = {spec.session_id: _load_session_frames(spec) for spec in specs}
    features: list[WindowFeatures] = []
    for window in windows:
        selected = [
            frame for frame in frames_by_session[window.session_id]
            if window.start_timestamp <= frame.timestamp < window.end_timestamp
        ]
        if len(selected) != window.frame_count:
            raise ValueError(
                f"window frame count mismatch for {window.window_id}: "
                f"manifest={window.frame_count}, raw={len(selected)}"
            )
        amplitude = np.stack([frame.amplitude for frame in selected])
        rssi = np.asarray([frame.rssi for frame in selected], dtype=np.float64)
        raw, normalized, nuisance, zero_count = extract_feature_vectors(amplitude, rssi)
        features.append(WindowFeatures(window, raw, normalized, nuisance, zero_count))
    return features


def make_group_folds(specs: Sequence[SessionSpec]) -> list[GroupFold]:
    session_ids = np.asarray([spec.session_id for spec in specs], dtype=object)
    labels = np.asarray([spec.label for spec in specs], dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    folds: list[GroupFold] = []
    for fold_number, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros((len(specs), 1)), labels, groups=session_ids), start=1
    ):
        train_sessions = tuple(session_ids[train_indices].tolist())
        test_sessions = tuple(session_ids[test_indices].tolist())
        if set(train_sessions) & set(test_sessions):
            raise RuntimeError(f"session leakage detected in fold {fold_number}")
        train_labels = {labels[index] for index in train_indices}
        test_labels = {labels[index] for index in test_indices}
        if train_labels != set(LABELS) or test_labels != set(LABELS):
            raise RuntimeError(f"fold {fold_number} does not contain all labels")
        test_counts = Counter(labels[index] for index in test_indices)
        if any(test_counts[label] != 2 for label in LABELS):
            raise RuntimeError(f"fold {fold_number} is not 2 sessions per label: {test_counts}")
        folds.append(GroupFold(fold_number, train_sessions, test_sessions))
    if Counter(session for fold in folds for session in fold.test_sessions) != Counter(session_ids):
        raise RuntimeError("each session must appear in exactly one test fold")
    return folds


def build_model(model_name: str) -> Any:
    if model_name == "LOGISTIC_REGRESSION":
        return Pipeline((
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=3000,
                random_state=RANDOM_STATE,
            )),
        ))
    if model_name == "RANDOM_FOREST":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    raise ValueError(f"unknown model: {model_name}")


def fit_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Fit using train data only; test arrays are deliberately not accepted."""

    model = build_model(model_name)
    model.fit(x_train, y_train)
    return model


def _aligned_probabilities(model: Any, values: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(values)
    classes = list(model.classes_)
    return np.column_stack([probabilities[:, classes.index(label)] for label in LABELS])


def _metric_row(
    fold: int | str,
    feature_set: str,
    model_name: str,
    level: str,
    actual: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=LABELS, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=LABELS, average="macro", zero_division=0
    )
    row: dict[str, Any] = {
        "fold": fold,
        "feature_set": feature_set,
        "model": model_name,
        "level": level,
        "samples": len(actual),
        "accuracy": accuracy_score(actual, predicted),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }
    for index, label in enumerate(LABELS):
        prefix = label.lower()
        row[f"{prefix}_precision"] = precision[index]
        row[f"{prefix}_recall"] = recall[index]
        row[f"{prefix}_f1"] = f1[index]
        row[f"{prefix}_support"] = int(support[index])
    return row


def _confusion_rows(
    fold: int | str,
    feature_set: str,
    model_name: str,
    level: str,
    actual: Sequence[str],
    predicted: Sequence[str],
) -> list[dict[str, Any]]:
    matrix = confusion_matrix(actual, predicted, labels=LABELS)
    return [
        {
            "fold": fold,
            "feature_set": feature_set,
            "model": model_name,
            "level": level,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "count": int(matrix[actual_index, predicted_index]),
        }
        for actual_index, actual_label in enumerate(LABELS)
        for predicted_index, predicted_label in enumerate(LABELS)
    ]


def _feature_matrix(
    features: Sequence[WindowFeatures], feature_set: str
) -> np.ndarray:
    attribute = {
        "RAW": "raw",
        "NORMALIZED": "normalized",
        "NUISANCE_ONLY": "nuisance",
    }[feature_set]
    return np.stack([getattr(feature, attribute) for feature in features])


def _aggregate_session_predictions(
    fold: int,
    feature_set: str,
    model_name: str,
    test_features: Sequence[WindowFeatures],
    probabilities: np.ndarray,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    grouped_indices: dict[str, list[int]] = defaultdict(list)
    for index, feature in enumerate(test_features):
        grouped_indices[feature.window.session_id].append(index)
    rows: list[dict[str, Any]] = []
    actual: list[str] = []
    predicted: list[str] = []
    for session_id in sorted(grouped_indices):
        indices = grouped_indices[session_id]
        mean_probability = probabilities[indices].mean(axis=0)
        prediction = LABELS[int(np.argmax(mean_probability))]
        label = test_features[indices[0]].window.label
        rows.append({
            "fold": fold,
            "feature_set": feature_set,
            "model": model_name,
            "session_id": session_id,
            "actual_label": label,
            "predicted_label": prediction,
            "correct": label == prediction,
            "window_count": len(indices),
            "aggregation_method": SESSION_PREDICTION_METHOD,
            "prob_sitting": mean_probability[LABELS.index("SITTING")],
            "prob_moving": mean_probability[LABELS.index("MOVING")],
            "prob_lying": mean_probability[LABELS.index("LYING")],
        })
        actual.append(label)
        predicted.append(prediction)
    return rows, actual, predicted


def run_cross_validation(
    features: Sequence[WindowFeatures], folds: Sequence[GroupFold]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_metrics: list[dict[str, Any]] = []
    session_predictions: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    all_predictions: dict[tuple[str, str, str], tuple[list[str], list[str]]] = defaultdict(
        lambda: ([], [])
    )

    for fold in folds:
        train_set = set(fold.train_sessions)
        test_set = set(fold.test_sessions)
        train_features = [f for f in features if f.window.session_id in train_set]
        test_features = [f for f in features if f.window.session_id in test_set]
        if {f.window.session_id for f in train_features} & {f.window.session_id for f in test_features}:
            raise RuntimeError(f"window session leakage detected in fold {fold.fold}")
        y_train = np.asarray([feature.window.label for feature in train_features], dtype=object)
        y_test = np.asarray([feature.window.label for feature in test_features], dtype=object)

        for feature_set in FEATURE_SETS:
            x_train = _feature_matrix(train_features, feature_set)
            x_test = _feature_matrix(test_features, feature_set)
            for model_name in MODEL_NAMES:
                model = fit_model(model_name, x_train, y_train)
                probabilities = _aligned_probabilities(model, x_test)
                predictions = np.asarray(
                    [LABELS[index] for index in np.argmax(probabilities, axis=1)], dtype=object
                )
                fold_metrics.append(_metric_row(
                    fold.fold, feature_set, model_name, "window", y_test, predictions
                ))
                confusion_rows.extend(_confusion_rows(
                    fold.fold, feature_set, model_name, "window", y_test, predictions
                ))
                all_window_actual, all_window_predicted = all_predictions[
                    (feature_set, model_name, "window")
                ]
                all_window_actual.extend(y_test.tolist())
                all_window_predicted.extend(predictions.tolist())

                session_rows, session_actual, session_predicted = _aggregate_session_predictions(
                    fold.fold, feature_set, model_name, test_features, probabilities
                )
                session_predictions.extend(session_rows)
                fold_metrics.append(_metric_row(
                    fold.fold, feature_set, model_name, "session",
                    session_actual, session_predicted,
                ))
                confusion_rows.extend(_confusion_rows(
                    fold.fold, feature_set, model_name, "session",
                    session_actual, session_predicted,
                ))
                all_session_actual, all_session_predicted = all_predictions[
                    (feature_set, model_name, "session")
                ]
                all_session_actual.extend(session_actual)
                all_session_predicted.extend(session_predicted)

    for (feature_set, model_name, level), (actual, predicted) in all_predictions.items():
        confusion_rows.extend(_confusion_rows(
            "ALL", feature_set, model_name, level, actual, predicted
        ))
    return fold_metrics, session_predictions, confusion_rows


def build_summary_rows(fold_metrics: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
    rows: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        for model_name in MODEL_NAMES:
            for level in ("window", "session"):
                selected = [
                    row for row in fold_metrics
                    if row["feature_set"] == feature_set
                    and row["model"] == model_name
                    and row["level"] == level
                ]
                output: dict[str, Any] = {
                    "feature_set": feature_set,
                    "model": model_name,
                    "level": level,
                    "folds": len(selected),
                    "random_state": RANDOM_STATE,
                    "session_prediction_method": (
                        SESSION_PREDICTION_METHOD if level == "session" else "not_applicable"
                    ),
                    "rssi_included": feature_set == "NUISANCE_ONLY",
                    "pca_used": False,
                    "experiment_scope": "within_subject_within_environment_session_level_baseline",
                    "numpy_version": np.__version__,
                    "sklearn_version": sklearn.__version__,
                }
                for metric_name in metric_names:
                    values = np.asarray([row[metric_name] for row in selected], dtype=np.float64)
                    output[f"{metric_name}_mean"] = values.mean()
                    output[f"{metric_name}_std"] = values.std()
                rows.append(output)
    return rows


def _format_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.9f}"
    if isinstance(value, np.integer):
        return int(value)
    return value


def _atomic_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format_value(row.get(column)) for column in columns})
    temporary.replace(path)


def _feature_rows(
    features: Sequence[WindowFeatures], attribute: str, names: Sequence[str]
) -> Iterable[dict[str, Any]]:
    for feature in features:
        window = feature.window
        row: dict[str, Any] = {
            "window_id": window.window_id,
            "label": window.label,
            "session_id": window.session_id,
            "group_id": window.group_id,
            "window_index": window.window_index,
            "start_timestamp": window.start_timestamp,
            "end_timestamp": window.end_timestamp,
            "duration": window.duration,
            "frame_count": window.frame_count,
            "zero_mean_frame_count": feature.zero_mean_frame_count,
        }
        row.update(zip(names, getattr(feature, attribute)))
        yield row


def write_outputs(result: BaselineResult, specs: Sequence[SessionSpec]) -> None:
    session_labels = {spec.session_id: spec.label for spec in specs}
    window_counts = Counter(feature.window.session_id for feature in result.feature_rows)
    assignment_rows = (
        {
            "fold": fold.fold,
            "role": role,
            "label": session_labels[session_id],
            "session_id": session_id,
            "window_count": window_counts[session_id],
        }
        for fold in result.folds
        for role, sessions in (("train", fold.train_sessions), ("test", fold.test_sessions))
        for session_id in sessions
    )
    _atomic_csv(
        result.output_dir / "fold_assignments.csv",
        ("fold", "role", "label", "session_id", "window_count"),
        assignment_rows,
    )
    _atomic_csv(
        result.output_dir / "window_features_raw.csv",
        FEATURE_METADATA_COLUMNS + RAW_FEATURE_NAMES,
        _feature_rows(result.feature_rows, "raw", RAW_FEATURE_NAMES),
    )
    _atomic_csv(
        result.output_dir / "window_features_normalized.csv",
        FEATURE_METADATA_COLUMNS + RAW_FEATURE_NAMES,
        _feature_rows(result.feature_rows, "normalized", RAW_FEATURE_NAMES),
    )
    _atomic_csv(
        result.output_dir / "nuisance_features.csv",
        FEATURE_METADATA_COLUMNS + NUISANCE_FEATURE_NAMES,
        _feature_rows(result.feature_rows, "nuisance", NUISANCE_FEATURE_NAMES),
    )
    metric_columns = tuple(result.fold_metric_rows[0].keys())
    _atomic_csv(result.output_dir / "fold_metrics.csv", metric_columns, result.fold_metric_rows)
    prediction_columns = tuple(result.session_prediction_rows[0].keys())
    _atomic_csv(
        result.output_dir / "session_predictions.csv",
        prediction_columns,
        result.session_prediction_rows,
    )
    confusion_columns = tuple(result.confusion_rows[0].keys())
    _atomic_csv(
        result.output_dir / "confusion_matrices.csv", confusion_columns, result.confusion_rows
    )
    summary_columns = tuple(result.summary_rows[0].keys())
    _atomic_csv(result.output_dir / "baseline_summary.csv", summary_columns, result.summary_rows)


def run_baseline(
    activity_root: str | Path,
    windows_manifest: str | Path,
    output_dir: str | Path,
) -> BaselineResult:
    specs = baseline_session_specs(activity_root)
    windows, _ = load_baseline_windows(windows_manifest, specs)
    features = tuple(build_window_features(specs, windows))
    folds = tuple(make_group_folds(specs))
    fold_metrics, session_predictions, confusion_rows = run_cross_validation(features, folds)
    summary_rows = build_summary_rows(fold_metrics)
    result = BaselineResult(
        folds,
        features,
        tuple(fold_metrics),
        tuple(session_predictions),
        tuple(confusion_rows),
        tuple(summary_rows),
        Path(output_dir),
    )
    write_outputs(result, specs)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train fixed session-grouped 3-class baseline and confound ablation."
    )
    parser.add_argument("activity_root", type=Path)
    parser.add_argument(
        "--windows-manifest", type=Path,
        default=Path("data/processed/activity_v1/windows_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/processed/activity_v1/baseline"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_baseline(args.activity_root, args.windows_manifest, args.output_dir)
    print("Activity baseline: session-grouped 4-fold CV")
    print(f"Sessions: 24; windows: {len(result.feature_rows)}; random_state: {RANDOM_STATE}")
    for fold in result.folds:
        test_counts = Counter(
            feature.window.label for feature in result.feature_rows
            if feature.window.session_id in set(fold.test_sessions)
        )
        session_counts = Counter(
            next(feature.window.label for feature in result.feature_rows if feature.window.session_id == session)
            for session in fold.test_sessions
        )
        print(
            f"Fold {fold.fold}: train_sessions={len(fold.train_sessions)} "
            f"test_sessions={len(fold.test_sessions)} "
            f"test_session_labels={dict(session_counts)} test_windows={dict(test_counts)}"
        )
        print(f"  test: {', '.join(fold.test_sessions)}")
    print("Mean/std across four folds:")
    for row in result.summary_rows:
        print(
            f"  {row['feature_set']:<13} {row['model']:<19} {row['level']:<7} "
            f"accuracy={row['accuracy_mean']:.3f}+/-{row['accuracy_std']:.3f} "
            f"macro_f1={row['macro_f1_mean']:.3f}+/-{row['macro_f1_std']:.3f}"
        )
    print(f"Output: {result.output_dir}")
    print("Scope: within-subject / within-environment session-level baseline only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
