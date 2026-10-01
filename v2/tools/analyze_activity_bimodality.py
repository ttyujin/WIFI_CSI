#!/usr/bin/env python3
"""Diagnose observed frame-mean amplitude grouping in the fixed baseline.

The analysis is descriptive.  It does not infer packet types, CSI modes,
hardware modes, activity labels, or any other cause.  Raw recording JSONL is
opened read-only; only derived CSV files are written.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

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


MIN_STANDARDIZED_SEPARATION = 3.0
MIN_RELATIVE_CENTER_DISTANCE = 0.10
MIN_VARIANCE_REDUCTION = 0.80
MIN_BALANCED_GROUP_RATIO = 0.10
MIN_RARE_GROUP_FRAMES = 3
MODULI = (2, 4, 8)
GROUP_LOW = "LOW"
GROUP_HIGH = "HIGH"
GROUP_UNSEPARATED = "UNSEPARATED"
OUTPUT_FILENAMES = (
    "frame_groups.csv",
    "session_bimodality_summary.csv",
    "label_bimodality_summary.csv",
    "window_bimodality_summary.csv",
    "subcarrier_group_summary.csv",
)


@dataclass(frozen=True)
class RawFrame:
    timestamp: float
    server_tick: int
    sequence: int
    rssi: float
    amplitude_mean: float
    amplitude_std: float
    amplitude: tuple[float, ...]


@dataclass(frozen=True)
class GroupFit:
    status: str
    reason: str
    candidate_groups: tuple[str, ...]
    groups: tuple[str, ...]
    candidate_low_count: int
    candidate_high_count: int
    low_center: float | None
    high_center: float | None
    center_distance: float | None
    relative_center_distance: float | None
    separation_strength: float | None
    smaller_group_ratio: float
    variance_reduction: float

    @property
    def has_distinct_groups(self) -> bool:
        return self.status in {"SEPARATED", "RARE_DISTINCT"}


@dataclass(frozen=True)
class AnalyzedFrame:
    label: str
    session_id: str
    timestamp: float
    server_tick: int
    sequence: int
    rssi: float
    amplitude_mean: float
    amplitude_std: float
    candidate_group: str
    group: str
    sequence_gap_from_previous: int | None
    tick_gap_from_previous: int | None
    group_changed: bool | None


@dataclass
class SessionAnalysis:
    spec: SessionSpec
    grouping: GroupFit
    frames: list[AnalyzedFrame]
    summary: dict[str, Any]
    subcarrier_rows: list[dict[str, Any]]
    total_nonempty_lines: int
    frame_lines: int
    excluded_frames: int
    excluded_records: int
    parse_errors: int
    metadata_errors: int
    error_reasons: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class AnalysisResult:
    sessions: tuple[SessionAnalysis, ...]
    label_rows: tuple[dict[str, Any], ...]
    window_rows: tuple[dict[str, Any], ...]
    output_dir: Path


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    center = sum(values) / len(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))


def _frame_mean_std(amplitude: Sequence[float]) -> tuple[float, float]:
    center = sum(amplitude) / len(amplitude)
    spread = math.sqrt(sum((value - center) ** 2 for value in amplitude) / len(amplitude))
    return center, spread


def fit_frame_mean_groups(values: Sequence[float]) -> GroupFit:
    """Fit deterministic 1-D two-means, then independently accept/reject it."""

    if len(values) < 2:
        groups = tuple(GROUP_UNSEPARATED for _ in values)
        return GroupFit(
            "UNSEPARATED", "insufficient_frames", groups, groups, 0, 0,
            None, None, None, None, None, 0.0, 0.0,
        )
    ordered = sorted(values)
    low_center = ordered[len(ordered) // 4]
    high_center = ordered[(3 * len(ordered)) // 4]
    if low_center == high_center:
        groups = tuple(GROUP_UNSEPARATED for _ in values)
        return GroupFit(
            "UNSEPARATED", "degenerate_frame_means", groups, groups, 0, 0,
            low_center, high_center, 0.0, 0.0, 0.0, 0.0, 0.0,
        )

    candidate_groups: tuple[str, ...] = ()
    for _ in range(100):
        midpoint = (low_center + high_center) / 2.0
        candidate_groups = tuple(
            GROUP_LOW if value <= midpoint else GROUP_HIGH for value in values
        )
        low_values = [value for value, group in zip(values, candidate_groups) if group == GROUP_LOW]
        high_values = [value for value, group in zip(values, candidate_groups) if group == GROUP_HIGH]
        if not low_values or not high_values:
            groups = tuple(GROUP_UNSEPARATED for _ in values)
            return GroupFit(
                "UNSEPARATED", "empty_candidate_group", candidate_groups, groups,
                len(low_values), len(high_values), low_center, high_center, None,
                None, None, 0.0, 0.0,
            )
        new_low = sum(low_values) / len(low_values)
        new_high = sum(high_values) / len(high_values)
        if abs(new_low - low_center) < 1e-12 and abs(new_high - high_center) < 1e-12:
            low_center, high_center = new_low, new_high
            break
        low_center, high_center = new_low, new_high

    low_values = [value for value, group in zip(values, candidate_groups) if group == GROUP_LOW]
    high_values = [value for value, group in zip(values, candidate_groups) if group == GROUP_HIGH]
    within_sse = sum((value - low_center) ** 2 for value in low_values)
    within_sse += sum((value - high_center) ** 2 for value in high_values)
    overall_center = sum(values) / len(values)
    overall_sse = sum((value - overall_center) ** 2 for value in values)
    pooled_within_std = math.sqrt(within_sse / len(values))
    center_distance = high_center - low_center
    separation_strength = center_distance / max(pooled_within_std, 1e-12)
    relative_distance = center_distance / max(abs(overall_center), 1e-12)
    smaller_count = min(len(low_values), len(high_values))
    smaller_ratio = smaller_count / len(values)
    variance_reduction = 1.0 - within_sse / overall_sse if overall_sse > 0 else 0.0

    weak_reasons: list[str] = []
    if separation_strength < MIN_STANDARDIZED_SEPARATION:
        weak_reasons.append("weak_standardized_separation")
    if relative_distance < MIN_RELATIVE_CENTER_DISTANCE:
        weak_reasons.append("small_relative_center_distance")
    if variance_reduction < MIN_VARIANCE_REDUCTION:
        weak_reasons.append("weak_variance_reduction")
    if smaller_count < MIN_RARE_GROUP_FRAMES:
        weak_reasons.append("too_few_frames_in_smaller_group")

    if weak_reasons:
        status = "UNSEPARATED"
        reason = ";".join(weak_reasons)
        groups = tuple(GROUP_UNSEPARATED for _ in values)
    elif smaller_ratio < MIN_BALANCED_GROUP_RATIO:
        status = "RARE_DISTINCT"
        reason = "strong_value_separation_but_smaller_group_below_10_percent"
        groups = candidate_groups
    else:
        status = "SEPARATED"
        reason = "accepted_by_descriptive_grouping_heuristic"
        groups = candidate_groups

    return GroupFit(
        status, reason, candidate_groups, groups, len(low_values), len(high_values),
        low_center, high_center, center_distance, relative_distance,
        separation_strength, smaller_ratio, variance_reduction,
    )


def temporal_run_statistics(
    groups: Sequence[str], timestamps: Sequence[float]
) -> dict[str, Any]:
    paired = [
        (group, timestamp)
        for group, timestamp in zip(groups, timestamps)
        if group in {GROUP_LOW, GROUP_HIGH}
    ]
    if not paired:
        return {
            "group_transition_count": 0,
            "transition_rate": None,
            "run_count": 0,
            "average_run_length_frames": None,
            "longest_run_frames": 0,
            "average_run_duration_sec": None,
            "longest_run_duration_sec": None,
            "longest_run_start_offset_sec": None,
            "longest_run_end_offset_sec": None,
            "first_group": "",
            "last_group": "",
            "final_run_length_frames": 0,
            "last_transition_offset_sec": None,
            "first_half_transition_count": 0,
            "second_half_transition_count": 0,
        }

    runs: list[tuple[str, int, int]] = []
    run_start = 0
    for index in range(1, len(paired)):
        if paired[index][0] != paired[index - 1][0]:
            runs.append((paired[index - 1][0], run_start, index - 1))
            run_start = index
    runs.append((paired[-1][0], run_start, len(paired) - 1))
    lengths = [end - start + 1 for _, start, end in runs]
    durations = [paired[end][1] - paired[start][1] for _, start, end in runs]
    longest_index = max(range(len(runs)), key=lambda index: (lengths[index], -index))
    _, longest_start, longest_end = runs[longest_index]
    transition_offsets = [
        paired[index][1] - paired[0][1]
        for index in range(1, len(paired))
        if paired[index][0] != paired[index - 1][0]
    ]
    midpoint = (paired[0][1] + paired[-1][1]) / 2.0
    first_half = sum(
        paired[index][1] <= midpoint
        for index in range(1, len(paired))
        if paired[index][0] != paired[index - 1][0]
    )
    return {
        "group_transition_count": len(runs) - 1,
        "transition_rate": (len(runs) - 1) / (len(paired) - 1) if len(paired) > 1 else 0.0,
        "run_count": len(runs),
        "average_run_length_frames": sum(lengths) / len(lengths),
        "longest_run_frames": max(lengths),
        "average_run_duration_sec": sum(durations) / len(durations),
        "longest_run_duration_sec": max(durations),
        "longest_run_start_offset_sec": paired[longest_start][1] - paired[0][1],
        "longest_run_end_offset_sec": paired[longest_end][1] - paired[0][1],
        "first_group": paired[0][0],
        "last_group": paired[-1][0],
        "final_run_length_frames": lengths[-1],
        "last_transition_offset_sec": transition_offsets[-1] if transition_offsets else None,
        "first_half_transition_count": first_half,
        "second_half_transition_count": len(transition_offsets) - first_half,
    }


def categorical_cramers_v(pairs: Sequence[tuple[Any, Any]]) -> float | None:
    if not pairs:
        return None
    row_values = sorted({str(row) for row, _ in pairs})
    column_values = sorted({str(column) for _, column in pairs})
    if len(row_values) < 2 or len(column_values) < 2:
        return 0.0
    counts = Counter((str(row), str(column)) for row, column in pairs)
    row_totals = Counter(str(row) for row, _ in pairs)
    column_totals = Counter(str(column) for _, column in pairs)
    total = len(pairs)
    chi_squared = 0.0
    for row in row_values:
        for column in column_values:
            expected = row_totals[row] * column_totals[column] / total
            if expected:
                observed = counts[(row, column)]
                chi_squared += (observed - expected) ** 2 / expected
    denominator = total * min(len(row_values) - 1, len(column_values) - 1)
    return math.sqrt(chi_squared / denominator) if denominator else 0.0


def modulo_statistics(
    groups: Sequence[str], values: Sequence[int], modulus: int
) -> tuple[float | None, str]:
    pairs = [
        (group, value % modulus)
        for group, value in zip(groups, values)
        if group in {GROUP_LOW, GROUP_HIGH}
    ]
    detail = {
        group: [sum(pair == (group, residue) for pair in pairs) for residue in range(modulus)]
        for group in (GROUP_LOW, GROUP_HIGH)
    }
    return categorical_cramers_v(pairs), json.dumps(detail, separators=(",", ":"))


def _pearson(values_a: Sequence[float], values_b: Sequence[float]) -> float | None:
    if len(values_a) != len(values_b) or not values_a:
        return None
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    delta_a = [value - mean_a for value in values_a]
    delta_b = [value - mean_b for value in values_b]
    denominator = math.sqrt(
        sum(value * value for value in delta_a) * sum(value * value for value in delta_b)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(delta_a, delta_b)) / denominator


def _profile_statistics(
    spec: SessionSpec, raw_frames: Sequence[RawFrame], grouping: GroupFit
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    empty = {
        "raw_profile_correlation": None,
        "mean_scale_ratio_high_over_low": None,
        "normalized_profile_correlation": None,
        "raw_profile_best_scale_high_from_low": None,
        "raw_profile_scale_fit_relative_rmse": None,
        "normalized_profile_mae": None,
        "normalized_profile_rmse": None,
        "normalized_profile_max_abs_difference": None,
        "normalized_subcarrier_fraction_abs_diff_gt_0_1": None,
        "normalized_subcarrier_fraction_abs_diff_gt_0_25": None,
        "normalizable_low_frames": 0,
        "normalizable_high_frames": 0,
        "profile_observation": "insufficient_distinct_groups",
    }
    if not grouping.has_distinct_groups:
        rows = [
            {
                "label": spec.label,
                "session_id": spec.session_id,
                "grouping_status": grouping.status,
                "subcarrier_index": index,
                "low_raw_mean": "",
                "high_raw_mean": "",
                "raw_difference_high_minus_low": "",
                "raw_ratio_high_over_low": "",
                "low_frame_mean_normalized_mean": "",
                "high_frame_mean_normalized_mean": "",
                "normalized_difference_high_minus_low": "",
            }
            for index in range(EXPECTED_SUBCARRIERS)
        ]
        return empty, rows

    low_frames = [frame for frame, group in zip(raw_frames, grouping.groups) if group == GROUP_LOW]
    high_frames = [frame for frame, group in zip(raw_frames, grouping.groups) if group == GROUP_HIGH]
    low_raw = [
        sum(frame.amplitude[index] for frame in low_frames) / len(low_frames)
        for index in range(EXPECTED_SUBCARRIERS)
    ]
    high_raw = [
        sum(frame.amplitude[index] for frame in high_frames) / len(high_frames)
        for index in range(EXPECTED_SUBCARRIERS)
    ]
    normalizable_low = [frame for frame in low_frames if abs(frame.amplitude_mean) > 1e-12]
    normalizable_high = [frame for frame in high_frames if abs(frame.amplitude_mean) > 1e-12]
    low_normalized = [
        sum(frame.amplitude[index] / frame.amplitude_mean for frame in normalizable_low)
        / len(normalizable_low)
        for index in range(EXPECTED_SUBCARRIERS)
    ] if normalizable_low else []
    high_normalized = [
        sum(frame.amplitude[index] / frame.amplitude_mean for frame in normalizable_high)
        / len(normalizable_high)
        for index in range(EXPECTED_SUBCARRIERS)
    ] if normalizable_high else []
    normalized_differences = [
        high - low for low, high in zip(low_normalized, high_normalized)
    ]
    normalized_correlation = _pearson(low_normalized, high_normalized)
    normalized_mae = (
        sum(abs(value) for value in normalized_differences) / len(normalized_differences)
        if normalized_differences else None
    )
    normalized_rmse = (
        math.sqrt(sum(value * value for value in normalized_differences) / len(normalized_differences))
        if normalized_differences else None
    )
    max_normalized_difference = (
        max(abs(value) for value in normalized_differences) if normalized_differences else None
    )
    low_energy = sum(value * value for value in low_raw)
    best_scale = (
        sum(low * high for low, high in zip(low_raw, high_raw)) / low_energy
        if low_energy else None
    )
    high_rms = math.sqrt(sum(value * value for value in high_raw) / len(high_raw))
    scale_fit_relative_rmse = (
        math.sqrt(
            sum((high - best_scale * low) ** 2 for low, high in zip(low_raw, high_raw))
            / len(high_raw)
        ) / high_rms
        if best_scale is not None and high_rms else None
    )
    if normalized_correlation is None or normalized_rmse is None:
        observation = "normalization_comparison_unavailable"
    elif normalized_correlation >= 0.98 and normalized_rmse <= 0.10:
        observation = "profiles_remain_similar_after_frame_mean_normalization"
    else:
        observation = "normalized_profiles_retain_observable_difference"

    summary = {
        "raw_profile_correlation": _pearson(low_raw, high_raw),
        "mean_scale_ratio_high_over_low": (
            grouping.high_center / grouping.low_center
            if grouping.low_center and grouping.high_center is not None else None
        ),
        "normalized_profile_correlation": normalized_correlation,
        "raw_profile_best_scale_high_from_low": best_scale,
        "raw_profile_scale_fit_relative_rmse": scale_fit_relative_rmse,
        "normalized_profile_mae": normalized_mae,
        "normalized_profile_rmse": normalized_rmse,
        "normalized_profile_max_abs_difference": max_normalized_difference,
        "normalized_subcarrier_fraction_abs_diff_gt_0_1": (
            sum(abs(value) > 0.10 for value in normalized_differences)
            / len(normalized_differences)
            if normalized_differences else None
        ),
        "normalized_subcarrier_fraction_abs_diff_gt_0_25": (
            sum(abs(value) > 0.25 for value in normalized_differences)
            / len(normalized_differences)
            if normalized_differences else None
        ),
        "normalizable_low_frames": len(normalizable_low),
        "normalizable_high_frames": len(normalizable_high),
        "profile_observation": observation,
    }
    rows = []
    for index in range(EXPECTED_SUBCARRIERS):
        low_value = low_raw[index]
        high_value = high_raw[index]
        rows.append({
            "label": spec.label,
            "session_id": spec.session_id,
            "grouping_status": grouping.status,
            "subcarrier_index": index,
            "low_raw_mean": low_value,
            "high_raw_mean": high_value,
            "raw_difference_high_minus_low": high_value - low_value,
            "raw_ratio_high_over_low": high_value / low_value if abs(low_value) > 1e-12 else None,
            "low_frame_mean_normalized_mean": low_normalized[index] if low_normalized else None,
            "high_frame_mean_normalized_mean": high_normalized[index] if high_normalized else None,
            "normalized_difference_high_minus_low": (
                normalized_differences[index] if normalized_differences else None
            ),
        })
    return summary, rows


def _gap_statistics(frames: Sequence[RawFrame], groups: Sequence[str], attribute: str) -> dict[str, Any]:
    gap_events = 0
    changes_after_gap = 0
    non_gap_pairs = 0
    changes_without_gap = 0
    gaps: list[int | None] = [None]
    for previous, current, previous_group, current_group in zip(
        frames, frames[1:], groups, groups[1:]
    ):
        delta = int(getattr(current, attribute)) - int(getattr(previous, attribute))
        gap = delta - 1
        gaps.append(gap)
        changed = (
            previous_group in {GROUP_LOW, GROUP_HIGH}
            and current_group in {GROUP_LOW, GROUP_HIGH}
            and previous_group != current_group
        )
        if gap != 0:
            gap_events += 1
            changes_after_gap += int(changed)
        else:
            non_gap_pairs += 1
            changes_without_gap += int(changed)
    prefix = "sequence" if attribute == "sequence" else "tick"
    return {
        f"{prefix}_gaps": gaps,
        f"{prefix}_gap_event_count": gap_events,
        f"group_changes_after_{prefix}_gap": changes_after_gap,
        f"group_change_rate_after_{prefix}_gap": (
            changes_after_gap / gap_events if gap_events else None
        ),
        f"group_change_rate_without_{prefix}_gap": (
            changes_without_gap / non_gap_pairs if non_gap_pairs else None
        ),
    }


def analyze_session(spec: SessionSpec) -> SessionAnalysis:
    raw_frames: list[RawFrame] = []
    total_nonempty_lines = 0
    frame_lines = 0
    excluded_frames = 0
    excluded_records = 0
    parse_errors = 0
    metadata_errors = 0
    metadata_count = 0
    error_reasons: Counter[str] = Counter()
    previous_timestamp: float | None = None

    if not spec.path.is_file():
        error_reasons["missing_file"] += 1
        grouping = fit_frame_mean_groups([])
        return SessionAnalysis(
            spec, grouping, [], {"audit_status": "FAIL", "session_id": spec.session_id}, [],
            0, 0, 0, 0, 0, 1, error_reasons,
        )

    with spec.path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            if not raw_line.strip():
                continue
            total_nonempty_lines += 1
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, ValueError):
                parse_errors += 1
                excluded_records += 1
                error_reasons["json_parse_error"] += 1
                continue
            if not isinstance(row, dict):
                parse_errors += 1
                excluded_records += 1
                error_reasons["json_object_required"] += 1
                continue
            if row.get("record_type") == "session":
                metadata_count += 1
                if row.get("label") != spec.label:
                    metadata_errors += 1
                    error_reasons["metadata_label_mismatch"] += 1
                if row.get("session_id") != spec.session_id:
                    metadata_errors += 1
                    error_reasons["metadata_session_id_mismatch"] += 1
                if row.get("source") != "esp32":
                    metadata_errors += 1
                    error_reasons["metadata_source_not_esp32"] += 1
                continue
            if row.get("record_type") != "frame":
                excluded_records += 1
                error_reasons["unknown_record_type"] += 1
                continue

            frame_lines += 1
            errors, amplitude = _validate_frame(row, spec)
            server_tick = _integer(row.get("server_tick"))
            if server_tick is None or server_tick < 0:
                errors.append("invalid_server_tick")
            timestamp = float(row["timestamp"]) if _finite_number(row.get("timestamp")) else None
            if timestamp is not None and previous_timestamp is not None and timestamp < previous_timestamp:
                errors.append("timestamp_inversion")
            if errors or amplitude is None:
                excluded_frames += 1
                error_reasons.update(errors or ["invalid_frame"])
                continue

            previous_timestamp = timestamp
            amplitude_mean, amplitude_std = _frame_mean_std(amplitude)
            raw_frames.append(RawFrame(
                timestamp=timestamp,
                server_tick=server_tick,
                sequence=int(row["sequence"]),
                rssi=float(row["rssi_dbm"]),
                amplitude_mean=amplitude_mean,
                amplitude_std=amplitude_std,
                amplitude=tuple(amplitude),
            ))

    if metadata_count != 1:
        metadata_errors += 1
        error_reasons[f"metadata_count_{metadata_count}"] += 1

    grouping = fit_frame_mean_groups([frame.amplitude_mean for frame in raw_frames])
    temporal = temporal_run_statistics(
        grouping.groups, [frame.timestamp for frame in raw_frames]
    )
    sequence_gaps = _gap_statistics(raw_frames, grouping.groups, "sequence")
    tick_gaps = _gap_statistics(raw_frames, grouping.groups, "server_tick")
    analyzed_frames: list[AnalyzedFrame] = []
    for index, (frame, candidate_group, group) in enumerate(
        zip(raw_frames, grouping.candidate_groups, grouping.groups)
    ):
        previous_group = grouping.groups[index - 1] if index else None
        group_changed = (
            group != previous_group
            if index and group in {GROUP_LOW, GROUP_HIGH}
            and previous_group in {GROUP_LOW, GROUP_HIGH}
            else None
        )
        analyzed_frames.append(AnalyzedFrame(
            spec.label, spec.session_id, frame.timestamp, frame.server_tick,
            frame.sequence, frame.rssi, frame.amplitude_mean, frame.amplitude_std,
            candidate_group, group,
            sequence_gaps["sequence_gaps"][index],
            tick_gaps["tick_gaps"][index],
            group_changed,
        ))

    low_frames = [
        frame for frame, group in zip(raw_frames, grouping.groups) if group == GROUP_LOW
    ]
    high_frames = [
        frame for frame, group in zip(raw_frames, grouping.groups) if group == GROUP_HIGH
    ]
    low_rssi = [frame.rssi for frame in low_frames]
    high_rssi = [frame.rssi for frame in high_frames]
    profile_summary, subcarrier_rows = _profile_statistics(spec, raw_frames, grouping)
    audit_status = (
        "FAIL" if metadata_errors or not raw_frames
        else "WARN" if excluded_frames or excluded_records or parse_errors
        else "PASS"
    )
    summary: dict[str, Any] = {
        "label": spec.label,
        "session_id": spec.session_id,
        "source_file": spec.source_file,
        "audit_status": audit_status,
        "grouping_status": grouping.status,
        "grouping_reason": grouping.reason,
        "total_frames": len(raw_frames),
        "low_group_frames": len(low_frames),
        "high_group_frames": len(high_frames),
        "unseparated_frames": len(raw_frames) - len(low_frames) - len(high_frames),
        "low_ratio": len(low_frames) / len(raw_frames) if raw_frames else None,
        "high_ratio": len(high_frames) / len(raw_frames) if raw_frames else None,
        "candidate_low_frames": grouping.candidate_low_count,
        "candidate_high_frames": grouping.candidate_high_count,
        "candidate_low_ratio": grouping.candidate_low_count / len(raw_frames) if raw_frames else None,
        "candidate_high_ratio": grouping.candidate_high_count / len(raw_frames) if raw_frames else None,
        "low_center": grouping.low_center,
        "high_center": grouping.high_center,
        "center_distance": grouping.center_distance,
        "relative_center_distance": grouping.relative_center_distance,
        "separation_strength": grouping.separation_strength,
        "smaller_group_ratio": grouping.smaller_group_ratio,
        "variance_reduction": grouping.variance_reduction,
        **temporal,
        "low_rssi_mean": _mean(low_rssi),
        "low_rssi_std": _std(low_rssi),
        "high_rssi_mean": _mean(high_rssi),
        "high_rssi_std": _std(high_rssi),
        "rssi_mean_difference_high_minus_low": (
            _mean(high_rssi) - _mean(low_rssi) if low_rssi and high_rssi else None
        ),
        "rssi_group_point_biserial_correlation": _pearson(
            [1.0 if group == GROUP_HIGH else 0.0 for group in grouping.groups],
            [frame.rssi for frame in raw_frames],
        ) if grouping.has_distinct_groups else None,
        "sequence_gap_event_count": sequence_gaps["sequence_gap_event_count"],
        "group_changes_after_sequence_gap": sequence_gaps["group_changes_after_sequence_gap"],
        "group_change_rate_after_sequence_gap": sequence_gaps["group_change_rate_after_sequence_gap"],
        "group_change_rate_without_sequence_gap": sequence_gaps["group_change_rate_without_sequence_gap"],
        "tick_gap_event_count": tick_gaps["tick_gap_event_count"],
        "group_changes_after_tick_gap": tick_gaps["group_changes_after_tick_gap"],
        "group_change_rate_after_tick_gap": tick_gaps["group_change_rate_after_tick_gap"],
        "group_change_rate_without_tick_gap": tick_gaps["group_change_rate_without_tick_gap"],
        **profile_summary,
        "total_nonempty_lines": total_nonempty_lines,
        "frame_lines": frame_lines,
        "excluded_frames": excluded_frames,
        "excluded_records": excluded_records,
        "parse_errors": parse_errors,
        "metadata_errors": metadata_errors,
        "error_reasons": json.dumps(dict(sorted(error_reasons.items())), separators=(",", ":")),
    }
    for modulus in MODULI:
        sequence_v, sequence_detail = modulo_statistics(
            grouping.groups, [frame.sequence for frame in raw_frames], modulus
        )
        tick_v, tick_detail = modulo_statistics(
            grouping.groups, [frame.server_tick for frame in raw_frames], modulus
        )
        summary[f"sequence_mod{modulus}_cramers_v"] = sequence_v
        summary[f"sequence_mod{modulus}_counts"] = sequence_detail
        summary[f"server_tick_mod{modulus}_cramers_v"] = tick_v
        summary[f"server_tick_mod{modulus}_counts"] = tick_detail

    return SessionAnalysis(
        spec, grouping, analyzed_frames, summary, subcarrier_rows,
        total_nonempty_lines, frame_lines, excluded_frames, excluded_records,
        parse_errors, metadata_errors, error_reasons,
    )


def build_label_rows(sessions: Sequence[SessionAnalysis]) -> list[dict[str, Any]]:
    all_pairs = [
        (frame.label, frame.group)
        for session in sessions for frame in session.frames
        if frame.group in {GROUP_LOW, GROUP_HIGH}
    ]
    global_label_v = categorical_cramers_v(all_pairs)
    rows: list[dict[str, Any]] = []
    for label in LABELS:
        selected = [session for session in sessions if session.spec.label == label]
        frames = [frame for session in selected for frame in session.frames]
        classified = [frame for frame in frames if frame.group in {GROUP_LOW, GROUP_HIGH}]
        low_count = sum(frame.group == GROUP_LOW for frame in classified)
        high_count = sum(frame.group == GROUP_HIGH for frame in classified)
        candidate_low = sum(frame.candidate_group == GROUP_LOW for frame in frames)
        candidate_high = sum(frame.candidate_group == GROUP_HIGH for frame in frames)
        row: dict[str, Any] = {
            "label": label,
            "sessions": len(selected),
            "separated_sessions": sum(session.grouping.status == "SEPARATED" for session in selected),
            "rare_distinct_sessions": sum(session.grouping.status == "RARE_DISTINCT" for session in selected),
            "unseparated_sessions": sum(session.grouping.status == "UNSEPARATED" for session in selected),
            "total_frames": len(frames),
            "classified_frames": len(classified),
            "unseparated_frames": len(frames) - len(classified),
            "low_group_frames": low_count,
            "high_group_frames": high_count,
            "low_ratio_among_classified": low_count / len(classified) if classified else None,
            "high_ratio_among_classified": high_count / len(classified) if classified else None,
            "low_ratio_all_frames": low_count / len(frames) if frames else None,
            "high_ratio_all_frames": high_count / len(frames) if frames else None,
            "candidate_low_ratio": candidate_low / len(frames) if frames else None,
            "candidate_high_ratio": candidate_high / len(frames) if frames else None,
            "label_vs_group_cramers_v_all_labels": global_label_v,
            "rssi_group_point_biserial_correlation": _pearson(
                [1.0 if frame.group == GROUP_HIGH else 0.0 for frame in classified],
                [frame.rssi for frame in classified],
            ),
        }
        for field_name, source_attribute in (
            ("sequence_mod2_cramers_v", "sequence"),
            ("sequence_mod4_cramers_v", "sequence"),
            ("sequence_mod8_cramers_v", "sequence"),
            ("server_tick_mod2_cramers_v", "server_tick"),
            ("server_tick_mod4_cramers_v", "server_tick"),
            ("server_tick_mod8_cramers_v", "server_tick"),
        ):
            modulus = int(field_name.split("mod", 1)[1].split("_", 1)[0])
            pairs = [
                (frame.group, getattr(frame, source_attribute) % modulus)
                for frame in classified
            ]
            row[field_name] = categorical_cramers_v(pairs)
        rows.append(row)
    return rows


def read_window_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def build_window_rows(
    manifest_rows: Sequence[dict[str, str]], sessions: Sequence[SessionAnalysis]
) -> list[dict[str, Any]]:
    session_map = {session.spec.session_id: session for session in sessions}
    session_order = {session.spec.session_id: index for index, session in enumerate(sessions)}
    rows: list[dict[str, Any]] = []
    for manifest in manifest_rows:
        session_id = manifest.get("session_id", "")
        if session_id not in session_map:
            continue
        if manifest.get("group_id") not in {None, "", session_id}:
            raise ValueError(f"window group_id crosses session boundary: {session_id}")
        session = session_map[session_id]
        if manifest.get("label") != session.spec.label:
            raise ValueError(f"window label/session mismatch: {session_id}")
        start = float(manifest["start_timestamp"])
        end = float(manifest["end_timestamp"])
        selected = [frame for frame in session.frames if start <= frame.timestamp < end]
        low_count = sum(frame.group == GROUP_LOW for frame in selected)
        high_count = sum(frame.group == GROUP_HIGH for frame in selected)
        candidate_low = sum(frame.candidate_group == GROUP_LOW for frame in selected)
        candidate_high = sum(frame.candidate_group == GROUP_HIGH for frame in selected)
        transition_count = sum(
            current.group != previous.group
            for previous, current in zip(selected, selected[1:])
            if previous.group in {GROUP_LOW, GROUP_HIGH}
            and current.group in {GROUP_LOW, GROUP_HIGH}
        )
        rows.append({
            **manifest,
            "grouping_status": session.grouping.status,
            "diagnostic_frame_count": len(selected),
            "low_frame_count": low_count,
            "high_frame_count": high_count,
            "unseparated_frame_count": len(selected) - low_count - high_count,
            "low_frame_ratio": low_count / len(selected) if selected else None,
            "high_frame_ratio": high_count / len(selected) if selected else None,
            "candidate_low_frame_ratio": candidate_low / len(selected) if selected else None,
            "candidate_high_frame_ratio": candidate_high / len(selected) if selected else None,
            "group_transition_count": transition_count,
        })
    rows.sort(key=lambda row: (session_order[row["session_id"]], int(row["window_index"])))
    return rows


def _format_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.9f}"
    return value


def _atomic_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in columns})
    temporary.replace(path)


def write_outputs(result: AnalysisResult) -> None:
    frame_columns = (
        "label", "session_id", "timestamp", "server_tick", "sequence", "rssi_dbm",
        "amplitude_mean", "amplitude_std", "grouping_status", "candidate_group", "group",
        "sequence_gap_from_previous", "tick_gap_from_previous", "group_changed",
    )
    frame_rows = (
        {
            "label": frame.label,
            "session_id": frame.session_id,
            "timestamp": frame.timestamp,
            "server_tick": frame.server_tick,
            "sequence": frame.sequence,
            "rssi_dbm": frame.rssi,
            "amplitude_mean": frame.amplitude_mean,
            "amplitude_std": frame.amplitude_std,
            "grouping_status": session.grouping.status,
            "candidate_group": frame.candidate_group,
            "group": frame.group,
            "sequence_gap_from_previous": frame.sequence_gap_from_previous,
            "tick_gap_from_previous": frame.tick_gap_from_previous,
            "group_changed": frame.group_changed,
        }
        for session in result.sessions for frame in session.frames
    )
    _atomic_csv(result.output_dir / "frame_groups.csv", frame_columns, frame_rows)

    session_columns = tuple(dict.fromkeys(
        key for session in result.sessions for key in session.summary
    ))
    _atomic_csv(
        result.output_dir / "session_bimodality_summary.csv",
        session_columns,
        (session.summary for session in result.sessions),
    )
    label_columns = tuple(result.label_rows[0].keys()) if result.label_rows else ()
    _atomic_csv(
        result.output_dir / "label_bimodality_summary.csv", label_columns, result.label_rows
    )
    window_columns = tuple(result.window_rows[0].keys()) if result.window_rows else ()
    _atomic_csv(
        result.output_dir / "window_bimodality_summary.csv", window_columns, result.window_rows
    )
    subcarrier_columns = (
        "label", "session_id", "grouping_status", "subcarrier_index", "low_raw_mean",
        "high_raw_mean", "raw_difference_high_minus_low", "raw_ratio_high_over_low",
        "low_frame_mean_normalized_mean", "high_frame_mean_normalized_mean",
        "normalized_difference_high_minus_low",
    )
    _atomic_csv(
        result.output_dir / "subcarrier_group_summary.csv",
        subcarrier_columns,
        (row for session in result.sessions for row in session.subcarrier_rows),
    )


def analyze_dataset(
    activity_root: str | Path,
    windows_manifest: str | Path,
    output_dir: str | Path,
) -> AnalysisResult:
    sessions = tuple(analyze_session(spec) for spec in baseline_session_specs(activity_root))
    label_rows = tuple(build_label_rows(sessions))
    window_rows = tuple(build_window_rows(read_window_manifest(windows_manifest), sessions))
    result = AnalysisResult(sessions, label_rows, window_rows, Path(output_dir))
    write_outputs(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose observed amplitude grouping without inferring its cause."
    )
    parser.add_argument("activity_root", type=Path)
    parser.add_argument(
        "--windows-manifest", type=Path,
        default=Path("data/processed/activity_v1/windows_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/processed/activity_v1/bimodality"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze_dataset(args.activity_root, args.windows_manifest, args.output_dir)
    print("Activity amplitude grouping diagnostic (descriptive; cause not inferred)")
    print(f"Sessions           : {len(result.sessions)}")
    print(f"Valid frames       : {sum(len(session.frames) for session in result.sessions)}")
    print(f"Excluded frames    : {sum(session.excluded_frames for session in result.sessions)}")
    for status in ("SEPARATED", "RARE_DISTINCT", "UNSEPARATED"):
        print(f"{status:<19}: {sum(session.grouping.status == status for session in result.sessions)}")
    print("Label low/high ratios among frames with an accepted distinct grouping:")
    for row in result.label_rows:
        print(
            f"  {row['label']:<7}: low={_format_value(row['low_ratio_among_classified'])} "
            f"high={_format_value(row['high_ratio_among_classified'])} "
            f"unseparated_sessions={row['unseparated_sessions']}"
        )
    print(f"Windows            : {len(result.window_rows)}")
    print(f"Output directory   : {result.output_dir}")
    print("Cause              : unknown (data-only diagnostic)")
    return 1 if any(session.summary.get("audit_status") == "FAIL" for session in result.sessions) else 0


if __name__ == "__main__":
    raise SystemExit(main())
