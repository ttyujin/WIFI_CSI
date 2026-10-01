#!/usr/bin/env python3
"""Audit and timestamp-window the explicit 24-session activity baseline.

This tool never writes to ``data/recordings/activity``.  It validates raw
ADR-018 frame fields, reports amplitude distributions, and writes derived CSV
metadata only.  It does not train a model and never consumes RuView persons,
keypoints, person_count, motion_level, or other inferred targets.
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


LABELS = ("SITTING", "MOVING", "LYING")
EXPECTED_SUBCARRIERS = 306
DEFAULT_WINDOW_SECONDS = 3.0
DEFAULT_HOP_SECONDS = 1.5
OUTPUT_FILENAMES = (
    "windows_manifest.csv",
    "session_summary.csv",
    "amplitude_audit.csv",
)


@dataclass(frozen=True)
class SessionSpec:
    label: str
    session_id: str
    path: Path
    source_file: str


@dataclass(frozen=True)
class FramePoint:
    timestamp: float
    rssi: float
    node_id: int
    sequence: int


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.count) if self.count else 0.0

    def values(self) -> tuple[int, float | None, float | None, float | None, float | None]:
        if not self.count:
            return (0, None, None, None, None)
        return (self.count, self.mean, self.std, self.minimum, self.maximum)


@dataclass(frozen=True)
class FrameGroupSummary:
    low_count: int
    high_count: int
    low_center: float | None
    high_center: float | None
    center_separation: float | None
    standardized_separation: float | None
    smaller_group_fraction: float
    variance_reduction: float
    unusually_distinct_groups: bool


@dataclass
class SessionAudit:
    spec: SessionSpec
    status: str
    frames: list[FramePoint] = field(default_factory=list)
    total_nonempty_lines: int = 0
    frame_lines: int = 0
    valid_frames: int = 0
    excluded_frames: int = 0
    excluded_records: int = 0
    parse_errors: int = 0
    metadata_errors: int = 0
    frame_errors: int = 0
    error_reasons: Counter[str] = field(default_factory=Counter)
    node_ids: set[int] = field(default_factory=set)
    rssi_stats: RunningStats = field(default_factory=RunningStats)
    amplitude_stats: RunningStats = field(default_factory=RunningStats)
    frame_mean_stats: RunningStats = field(default_factory=RunningStats)
    frame_std_stats: RunningStats = field(default_factory=RunningStats)
    subcarrier_stats: list[RunningStats] = field(
        default_factory=lambda: [RunningStats() for _ in range(EXPECTED_SUBCARRIERS)]
    )
    frame_means: list[float] = field(default_factory=list)
    group_summary: FrameGroupSummary | None = None

    @property
    def first_timestamp(self) -> float | None:
        return self.frames[0].timestamp if self.frames else None

    @property
    def last_timestamp(self) -> float | None:
        return self.frames[-1].timestamp if self.frames else None

    @property
    def duration_sec(self) -> float | None:
        if len(self.frames) < 2:
            return 0.0 if self.frames else None
        return self.frames[-1].timestamp - self.frames[0].timestamp

    @property
    def estimated_fps(self) -> float | None:
        duration = self.duration_sec
        if duration is None or duration <= 0 or len(self.frames) < 2:
            return None
        return (len(self.frames) - 1) / duration


@dataclass(frozen=True)
class WindowRecord:
    label: str
    session_id: str
    window_index: int
    source_file: str
    start_timestamp: float
    end_timestamp: float
    duration: float
    frame_count: int
    mean_fps: float
    mean_rssi: float
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class PreparationResult:
    audits: tuple[SessionAudit, ...]
    windows: tuple[WindowRecord, ...]
    excluded_files: tuple[str, ...]
    output_dir: Path


def baseline_session_specs(activity_root: str | Path) -> list[SessionSpec]:
    """Return the fixed baseline allowlist; no directory discovery selects data."""

    root = Path(activity_root)
    specs: list[SessionSpec] = []
    for label in LABELS:
        prefix = label.lower()
        for number in range(1, 9):
            session_id = f"{prefix}_{number:03d}"
            relative = Path(label) / f"{session_id}.jsonl"
            specs.append(SessionSpec(label, session_id, root / relative, relative.as_posix()))
    return specs


def discover_excluded_files(activity_root: str | Path, specs: Sequence[SessionSpec]) -> list[str]:
    root = Path(activity_root)
    selected = {spec.path.resolve() for spec in specs}
    return [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*.jsonl"))
        if path.is_file() and path.resolve() not in selected
    ]


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _integer(value: Any) -> int | None:
    if not _finite_number(value) or not float(value).is_integer():
        return None
    return int(value)


def _validate_frame(row: dict[str, Any], spec: SessionSpec) -> tuple[list[str], list[float] | None]:
    errors: list[str] = []
    if row.get("label") != spec.label:
        errors.append("label_mismatch")
    if row.get("session_id") != spec.session_id:
        errors.append("session_id_mismatch")
    if row.get("source") != "esp32":
        errors.append("source_not_esp32")

    node_id = _integer(row.get("node_id"))
    if node_id is None:
        errors.append("invalid_node_id")
    timestamp = row.get("timestamp")
    if not _finite_number(timestamp):
        errors.append("invalid_timestamp")
    sequence = _integer(row.get("sequence"))
    if sequence is None or sequence < 0:
        errors.append("invalid_sequence")
    if not _finite_number(row.get("rssi_dbm")):
        errors.append("invalid_rssi")
    if _integer(row.get("subcarrier_count")) != EXPECTED_SUBCARRIERS:
        errors.append("subcarrier_count_not_306")
    if _integer(row.get("amplitude_count")) != EXPECTED_SUBCARRIERS:
        errors.append("amplitude_count_not_306")

    raw_amplitude = row.get("amplitude")
    amplitude: list[float] | None = None
    if not isinstance(raw_amplitude, list):
        errors.append("amplitude_not_array")
    elif len(raw_amplitude) != EXPECTED_SUBCARRIERS:
        errors.append("amplitude_length_not_306")
    elif any(not _finite_number(value) for value in raw_amplitude):
        errors.append("invalid_amplitude_value")
    else:
        amplitude = [float(value) for value in raw_amplitude]
    return errors, amplitude


def _distribution(values: Sequence[float]) -> RunningStats:
    stats = RunningStats()
    for value in values:
        stats.add(value)
    return stats


def summarize_frame_groups(values: Sequence[float]) -> FrameGroupSummary:
    """Describe two possible 1-D frame-mean groups without assigning a cause."""

    if len(values) < 2:
        return FrameGroupSummary(0, 0, None, None, None, None, 0.0, 0.0, False)
    ordered = sorted(values)
    low_center = ordered[len(ordered) // 4]
    high_center = ordered[(3 * len(ordered)) // 4]
    if low_center == high_center:
        return FrameGroupSummary(len(values), 0, low_center, high_center, 0.0, 0.0, 0.0, 0.0, False)

    low_group: list[float] = []
    high_group: list[float] = []
    for _ in range(50):
        midpoint = (low_center + high_center) / 2.0
        low_group = [value for value in values if value <= midpoint]
        high_group = [value for value in values if value > midpoint]
        if not low_group or not high_group:
            break
        new_low = sum(low_group) / len(low_group)
        new_high = sum(high_group) / len(high_group)
        if abs(new_low - low_center) < 1e-12 and abs(new_high - high_center) < 1e-12:
            break
        low_center, high_center = new_low, new_high

    if not low_group or not high_group:
        return FrameGroupSummary(len(values), 0, low_center, high_center, None, None, 0.0, 0.0, False)
    low_center = sum(low_group) / len(low_group)
    high_center = sum(high_group) / len(high_group)
    within_sse = sum((value - low_center) ** 2 for value in low_group)
    within_sse += sum((value - high_center) ** 2 for value in high_group)
    overall_center = sum(values) / len(values)
    overall_sse = sum((value - overall_center) ** 2 for value in values)
    pooled_std = math.sqrt(within_sse / len(values))
    separation = high_center - low_center
    standardized = separation / max(pooled_std, 1e-12)
    smaller_fraction = min(len(low_group), len(high_group)) / len(values)
    variance_reduction = 1.0 - within_sse / overall_sse if overall_sse > 0 else 0.0
    # Collection-audit heuristic only: this flags an observable separation in
    # frame means. It deliberately does not name or infer a packet type.
    distinct = standardized >= 3.0 and smaller_fraction >= 0.10
    return FrameGroupSummary(
        len(low_group),
        len(high_group),
        low_center,
        high_center,
        separation,
        standardized,
        smaller_fraction,
        variance_reduction,
        distinct,
    )


def read_and_audit_session(spec: SessionSpec) -> SessionAudit:
    audit = SessionAudit(spec=spec, status="FAIL")
    if not spec.path.is_file():
        audit.metadata_errors = 1
        audit.error_reasons["missing_file"] += 1
        return audit

    metadata_count = 0
    previous_timestamp: float | None = None
    with spec.path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            audit.total_nonempty_lines += 1
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, ValueError):
                audit.parse_errors += 1
                audit.excluded_records += 1
                audit.error_reasons["json_parse_error"] += 1
                continue
            if not isinstance(row, dict):
                audit.parse_errors += 1
                audit.excluded_records += 1
                audit.error_reasons["json_object_required"] += 1
                continue

            record_type = row.get("record_type")
            if record_type == "session":
                metadata_count += 1
                if row.get("session_id") != spec.session_id:
                    audit.metadata_errors += 1
                    audit.error_reasons["metadata_session_id_mismatch"] += 1
                if row.get("label") != spec.label:
                    audit.metadata_errors += 1
                    audit.error_reasons["metadata_label_mismatch"] += 1
                if row.get("source") != "esp32":
                    audit.metadata_errors += 1
                    audit.error_reasons["metadata_source_not_esp32"] += 1
                continue
            if record_type != "frame":
                audit.excluded_records += 1
                audit.error_reasons["unknown_record_type"] += 1
                continue

            audit.frame_lines += 1
            errors, amplitude = _validate_frame(row, spec)
            timestamp = float(row["timestamp"]) if _finite_number(row.get("timestamp")) else None
            if timestamp is not None and previous_timestamp is not None and timestamp < previous_timestamp:
                errors.append("timestamp_inversion")
            if errors or amplitude is None:
                audit.excluded_frames += 1
                audit.frame_errors += 1
                audit.error_reasons.update(errors or ["invalid_amplitude"])
                continue

            previous_timestamp = timestamp
            node_id = int(row["node_id"])
            sequence = int(row["sequence"])
            rssi = float(row["rssi_dbm"])
            audit.frames.append(FramePoint(timestamp, rssi, node_id, sequence))
            audit.valid_frames += 1
            audit.node_ids.add(node_id)
            audit.rssi_stats.add(rssi)

            frame_stats = _distribution(amplitude)
            audit.frame_means.append(frame_stats.mean)
            audit.frame_mean_stats.add(frame_stats.mean)
            audit.frame_std_stats.add(frame_stats.std)
            for index, value in enumerate(amplitude):
                audit.amplitude_stats.add(value)
                audit.subcarrier_stats[index].add(value)

    if metadata_count != 1:
        audit.metadata_errors += 1
        audit.error_reasons[f"metadata_count_{metadata_count}"] += 1
    audit.group_summary = summarize_frame_groups(audit.frame_means)
    if audit.metadata_errors or audit.valid_frames == 0:
        audit.status = "FAIL"
    elif audit.parse_errors or audit.frame_errors or audit.excluded_records:
        audit.status = "WARN"
    else:
        audit.status = "PASS"
    return audit


def build_timestamp_windows(
    audit: SessionAudit,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
) -> list[WindowRecord]:
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    if audit.status == "FAIL" or len(audit.frames) < 2:
        return []

    frames = audit.frames
    first_timestamp = frames[0].timestamp
    last_timestamp = frames[-1].timestamp
    windows: list[WindowRecord] = []
    left = 0
    right = 0
    window_index = 0
    epsilon = 1e-9
    while True:
        start = first_timestamp + window_index * hop_seconds
        end = start + window_seconds
        if end > last_timestamp + epsilon:
            break
        while left < len(frames) and frames[left].timestamp < start:
            left += 1
        right = max(right, left)
        while right < len(frames) and frames[right].timestamp < end:
            right += 1
        selected = frames[left:right]
        if selected:
            windows.append(
                WindowRecord(
                    label=audit.spec.label,
                    session_id=audit.spec.session_id,
                    window_index=window_index,
                    source_file=audit.spec.source_file,
                    start_timestamp=start,
                    end_timestamp=end,
                    duration=window_seconds,
                    frame_count=len(selected),
                    mean_fps=len(selected) / window_seconds,
                    mean_rssi=sum(frame.rssi for frame in selected) / len(selected),
                    node_ids=tuple(sorted({frame.node_id for frame in selected})),
                )
            )
        window_index += 1
    return windows


def _format_float(value: float | None, digits: int = 9) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _atomic_csv(
    path: Path,
    columns: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _window_rows(windows: Sequence[WindowRecord]) -> Iterable[dict[str, Any]]:
    for window in windows:
        yield {
            "label": window.label,
            "session_id": window.session_id,
            "window_index": window.window_index,
            "source_file": window.source_file,
            "start_timestamp": _format_float(window.start_timestamp, 6),
            "end_timestamp": _format_float(window.end_timestamp, 6),
            "duration": _format_float(window.duration, 3),
            "frame_count": window.frame_count,
            "mean_fps": _format_float(window.mean_fps, 6),
            "mean_rssi": _format_float(window.mean_rssi, 6),
            "node_id": "|".join(str(value) for value in window.node_ids),
            "group_id": window.session_id,
        }


def _session_rows(audits: Sequence[SessionAudit]) -> Iterable[dict[str, Any]]:
    for audit in audits:
        group = audit.group_summary
        yield {
            "label": audit.spec.label,
            "session_id": audit.spec.session_id,
            "dataset_role": "formal",
            "source_file": audit.spec.source_file,
            "audit_status": audit.status,
            "total_nonempty_lines": audit.total_nonempty_lines,
            "frame_lines": audit.frame_lines,
            "valid_frames": audit.valid_frames,
            "excluded_frames": audit.excluded_frames,
            "excluded_records": audit.excluded_records,
            "parse_errors": audit.parse_errors,
            "metadata_errors": audit.metadata_errors,
            "frame_errors": audit.frame_errors,
            "first_timestamp": _format_float(audit.first_timestamp, 6),
            "last_timestamp": _format_float(audit.last_timestamp, 6),
            "duration_sec": _format_float(audit.duration_sec, 6),
            "estimated_fps": _format_float(audit.estimated_fps, 6),
            "node_id": "|".join(str(value) for value in sorted(audit.node_ids)),
            "mean_rssi": _format_float(audit.rssi_stats.mean if audit.rssi_stats.count else None, 6),
            "amplitude_mean": _format_float(audit.amplitude_stats.mean if audit.amplitude_stats.count else None, 9),
            "amplitude_std": _format_float(audit.amplitude_stats.std if audit.amplitude_stats.count else None, 9),
            "frame_mean_mean": _format_float(audit.frame_mean_stats.mean if audit.frame_mean_stats.count else None, 9),
            "frame_mean_std": _format_float(audit.frame_mean_stats.std if audit.frame_mean_stats.count else None, 9),
            "frame_std_mean": _format_float(audit.frame_std_stats.mean if audit.frame_std_stats.count else None, 9),
            "frame_std_std": _format_float(audit.frame_std_stats.std if audit.frame_std_stats.count else None, 9),
            "distinct_frame_groups": str(bool(group and group.unusually_distinct_groups)).lower(),
            "group_low_center": _format_float(group.low_center if group else None, 9),
            "group_high_center": _format_float(group.high_center if group else None, 9),
            "group_standardized_separation": _format_float(group.standardized_separation if group else None, 9),
            "group_smaller_fraction": _format_float(group.smaller_group_fraction if group else None, 9),
            "group_variance_reduction": _format_float(group.variance_reduction if group else None, 9),
            "error_reasons": json.dumps(dict(sorted(audit.error_reasons.items())), separators=(",", ":")),
        }


def _amplitude_rows(audits: Sequence[SessionAudit]) -> Iterable[dict[str, Any]]:
    def row(
        audit: SessionAudit,
        metric: str,
        stats: RunningStats,
        subcarrier_index: int | str = "",
    ) -> dict[str, Any]:
        count, mean, std, minimum, maximum = stats.values()
        return {
            "label": audit.spec.label,
            "session_id": audit.spec.session_id,
            "metric": metric,
            "subcarrier_index": subcarrier_index,
            "count": count,
            "mean": _format_float(mean, 9),
            "std": _format_float(std, 9),
            "min": _format_float(minimum, 9),
            "max": _format_float(maximum, 9),
            "group_low_count": "",
            "group_high_count": "",
            "group_low_center": "",
            "group_high_center": "",
            "group_center_separation": "",
            "group_standardized_separation": "",
            "group_smaller_fraction": "",
            "group_variance_reduction": "",
            "unusually_distinct_frame_groups": "",
            "interpretation": "observed amplitude statistic; no packet-type inference",
        }

    for audit in audits:
        yield row(audit, "all_amplitudes", audit.amplitude_stats)
        yield row(audit, "frame_amplitude_mean_distribution", audit.frame_mean_stats)
        yield row(audit, "frame_amplitude_std_distribution", audit.frame_std_stats)
        for index, stats in enumerate(audit.subcarrier_stats):
            yield row(audit, "subcarrier_amplitude", stats, index)
        group = audit.group_summary
        if group:
            yield {
                "label": audit.spec.label,
                "session_id": audit.spec.session_id,
                "metric": "frame_mean_two_group_summary",
                "subcarrier_index": "",
                "count": len(audit.frame_means),
                "mean": "",
                "std": "",
                "min": "",
                "max": "",
                "group_low_count": group.low_count,
                "group_high_count": group.high_count,
                "group_low_center": _format_float(group.low_center, 9),
                "group_high_center": _format_float(group.high_center, 9),
                "group_center_separation": _format_float(group.center_separation, 9),
                "group_standardized_separation": _format_float(group.standardized_separation, 9),
                "group_smaller_fraction": _format_float(group.smaller_group_fraction, 9),
                "group_variance_reduction": _format_float(group.variance_reduction, 9),
                "unusually_distinct_frame_groups": str(group.unusually_distinct_groups).lower(),
                "interpretation": "observed frame-mean grouping heuristic; no packet-type inference",
            }


def write_outputs(result: PreparationResult) -> None:
    _atomic_csv(
        result.output_dir / "windows_manifest.csv",
        (
            "label", "session_id", "window_index", "source_file",
            "start_timestamp", "end_timestamp", "duration", "frame_count",
            "mean_fps", "mean_rssi", "node_id", "group_id",
        ),
        _window_rows(result.windows),
    )
    _atomic_csv(
        result.output_dir / "session_summary.csv",
        (
            "label", "session_id", "dataset_role", "source_file", "audit_status",
            "total_nonempty_lines", "frame_lines", "valid_frames", "excluded_frames",
            "excluded_records", "parse_errors", "metadata_errors", "frame_errors",
            "first_timestamp", "last_timestamp", "duration_sec", "estimated_fps",
            "node_id", "mean_rssi", "amplitude_mean", "amplitude_std",
            "frame_mean_mean", "frame_mean_std", "frame_std_mean", "frame_std_std",
            "distinct_frame_groups", "group_low_center", "group_high_center",
            "group_standardized_separation", "group_smaller_fraction",
            "group_variance_reduction", "error_reasons",
        ),
        _session_rows(result.audits),
    )
    _atomic_csv(
        result.output_dir / "amplitude_audit.csv",
        (
            "label", "session_id", "metric", "subcarrier_index", "count", "mean",
            "std", "min", "max", "group_low_count", "group_high_count",
            "group_low_center", "group_high_center", "group_center_separation",
            "group_standardized_separation", "group_smaller_fraction",
            "group_variance_reduction", "unusually_distinct_frame_groups", "interpretation",
        ),
        _amplitude_rows(result.audits),
    )


def prepare_dataset(
    activity_root: str | Path,
    output_dir: str | Path,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
) -> PreparationResult:
    if window_seconds <= 0 or hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    specs = baseline_session_specs(activity_root)
    audits = tuple(read_and_audit_session(spec) for spec in specs)
    windows = tuple(
        window
        for audit in audits
        for window in build_timestamp_windows(audit, window_seconds, hop_seconds)
    )
    result = PreparationResult(
        audits=audits,
        windows=windows,
        excluded_files=tuple(discover_excluded_files(activity_root, specs)),
        output_dir=Path(output_dir),
    )
    write_outputs(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and timestamp-window the fixed 24-session activity baseline; no model training."
    )
    parser.add_argument("activity_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/activity_v1"),
    )
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--hop-seconds", type=float, default=DEFAULT_HOP_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_dataset(
        args.activity_root,
        args.output_dir,
        args.window_seconds,
        args.hop_seconds,
    )
    print("Activity baseline preparation (audit + timestamp windows only)")
    print(f"Selected sessions : {len(result.audits)}")
    for label in LABELS:
        names = [audit.spec.session_id for audit in result.audits if audit.spec.label == label]
        print(f"  {label:<7}: {', '.join(names)}")
    print(f"Excluded JSONL    : {len(result.excluded_files)}")
    print(f"Valid frames      : {sum(audit.valid_frames for audit in result.audits)}")
    print(f"Excluded frames   : {sum(audit.excluded_frames for audit in result.audits)}")
    print(f"Parse errors      : {sum(audit.parse_errors for audit in result.audits)}")
    print(f"Windows           : {len(result.windows)}")
    for label in LABELS:
        print(f"  {label:<7}: {sum(window.label == label for window in result.windows)}")
    print("Distinct frame-mean groups (observational heuristic; not packet-type inference):")
    for audit in result.audits:
        group = audit.group_summary
        if group:
            print(
                f"  {audit.spec.session_id:<12}: {str(group.unusually_distinct_groups):<5} "
                f"centers={_format_float(group.low_center, 3)}/{_format_float(group.high_center, 3)} "
                f"separation={_format_float(group.standardized_separation, 3)}"
            )
    print(f"Output directory  : {result.output_dir}")
    for filename in OUTPUT_FILENAMES:
        print(f"  - {filename}")
    return 1 if any(audit.status == "FAIL" for audit in result.audits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
