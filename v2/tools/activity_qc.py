#!/usr/bin/env python3
"""Read-only quality checks for RuView activity CSI recordings.

The thresholds in this module are collection-QC heuristics for the current
ESP32 ADR-018 setup.  They are not learned decision boundaries, classifier
thresholds, or claims of research/clinical validity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ALLOWED_LABELS = ("SITTING", "MOVING", "LYING")
MANIFEST_COLUMNS = (
    "session_id",
    "label",
    "dataset_role",
    "file_path",
    "frames",
    "duration_sec",
    "fps",
    "node_id",
    "source",
    "mean_rssi",
    "min_rssi",
    "max_rssi",
    "first_sequence",
    "last_sequence",
    "sequence_gap_count",
    "sequence_gap_rate",
    "duplicate_sequences",
    "subcarrier_count",
    "amplitude_count",
    "amplitude_lengths",
    "invalid_amplitude_frames",
    "timestamps_monotonic",
    "qc_status",
    "qc_reasons",
)


@dataclass(frozen=True)
class QcHeuristics:
    """Collection-QC heuristics, not paper/model thresholds."""

    expected_subcarrier_count: int = 306
    expected_amplitude_count: int = 306
    expected_amplitude_length: int = 306
    expected_node_id: int = 1
    fail_min_duration_sec: float = 20.0
    duration_warn_min_sec: float = 25.0
    duration_warn_max_sec: float = 35.0
    fps_warn_min: float = 25.0
    fps_warn_max: float = 35.0
    sequence_gap_warn_rate: float = 0.05
    sequence_gap_strong_warn_rate: float = 0.10
    rssi_range_warn_db: float = 20.0
    name: str = "activity_collection_qc_heuristic_v2"


@dataclass(frozen=True)
class QcResult:
    file_path: str
    session_id: str
    label: str
    dataset_role: str
    frame_count: int
    duration_sec: float | None
    estimated_fps: float | None
    first_sequence: int | None
    last_sequence: int | None
    sequence_gap_count: int
    sequence_gap_rate: float
    duplicate_sequences: int
    sequence_reversals: int
    node_ids: tuple[int, ...]
    sources: tuple[str, ...]
    mean_rssi: float | None
    min_rssi: float | None
    max_rssi: float | None
    subcarrier_counts: tuple[int, ...]
    amplitude_counts: tuple[int, ...]
    amplitude_lengths: tuple[int, ...]
    invalid_amplitude_frames: int
    timestamp_inversions: int
    timestamps_monotonic: bool
    parse_error_count: int
    fail_reasons: tuple[str, ...]
    warn_reasons: tuple[str, ...]
    qc_status: str
    heuristic_name: str


def dataset_role(session_id: str) -> str:
    """Classify collection role from the session name only."""

    lowered = session_id.casefold()
    return "pilot" if "pilot" in lowered or "test" in lowered else "formal"


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)) or not float(value).is_integer():
        return None
    return int(value)


def _display_values(values: Sequence[Any]) -> str:
    return ",".join(str(value) for value in values) if values else "-"


def analyze_recording(
    file_path: str | Path,
    heuristics: QcHeuristics | None = None,
) -> QcResult:
    """Analyze one JSONL recording without modifying it."""

    rules = heuristics or QcHeuristics()
    path = Path(file_path)
    labels: set[str] = set()
    session_ids: set[str] = set()
    sources: set[str] = set()
    node_ids: set[int] = set()
    subcarrier_counts: set[int] = set()
    amplitude_counts: set[int] = set()
    amplitude_lengths: set[int] = set()
    timestamps: list[float] = []
    sequences: list[int] = []
    rssi_values: list[float] = []
    seen_sequences: set[int] = set()
    frame_count = 0
    metadata_count = 0
    parse_errors: list[str] = []
    structural_errors: list[str] = []
    warnings: list[str] = []
    invalid_amplitude_frames = 0
    invalid_rssi_frames = 0
    timestamp_inversions = 0
    duplicate_sequences = 0
    sequence_reversals = 0
    sequence_gap_count = 0
    previous_timestamp: float | None = None
    previous_forward_sequence: int | None = None

    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        parse_errors.append(f"file_read_error:{exc.__class__.__name__}")
        stream = None

    if stream is not None:
        with stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line)
                except (json.JSONDecodeError, ValueError) as exc:
                    parse_errors.append(
                        f"json_parse_error_line_{line_number}:{exc.__class__.__name__}"
                    )
                    continue
                if not isinstance(row, dict):
                    parse_errors.append(f"json_object_required_line_{line_number}")
                    continue

                record_type = row.get("record_type")
                label = row.get("label")
                session = row.get("session_id")
                source = row.get("source")
                if isinstance(label, str):
                    labels.add(label)
                if isinstance(session, str) and session:
                    session_ids.add(session)
                if isinstance(source, str):
                    sources.add(source)

                if record_type == "session":
                    metadata_count += 1
                    if not isinstance(label, str):
                        structural_errors.append(f"missing_metadata_label_line_{line_number}")
                    if not isinstance(session, str) or not session:
                        structural_errors.append(
                            f"missing_metadata_session_id_line_{line_number}"
                        )
                    if not isinstance(source, str):
                        structural_errors.append(f"missing_metadata_source_line_{line_number}")
                    continue
                if record_type != "frame":
                    structural_errors.append(f"unknown_record_type_line_{line_number}")
                    continue

                frame_count += 1
                if not isinstance(label, str):
                    structural_errors.append(f"missing_frame_label_line_{line_number}")
                if not isinstance(session, str) or not session:
                    structural_errors.append(f"missing_frame_session_id_line_{line_number}")
                if not isinstance(source, str):
                    structural_errors.append(f"missing_frame_source_line_{line_number}")

                node_id = _integer(row.get("node_id"))
                if node_id is None:
                    structural_errors.append(f"invalid_node_id_line_{line_number}")
                else:
                    node_ids.add(node_id)

                subcarrier_count = _integer(row.get("subcarrier_count"))
                if subcarrier_count is None:
                    structural_errors.append(f"invalid_subcarrier_count_line_{line_number}")
                else:
                    subcarrier_counts.add(subcarrier_count)

                amplitude_count = _integer(row.get("amplitude_count"))
                if amplitude_count is None:
                    structural_errors.append(f"invalid_amplitude_count_line_{line_number}")
                else:
                    amplitude_counts.add(amplitude_count)

                amplitude = row.get("amplitude")
                amplitude_invalid = not isinstance(amplitude, list)
                if isinstance(amplitude, list):
                    amplitude_lengths.add(len(amplitude))
                    amplitude_invalid = any(not _finite_number(value) for value in amplitude)
                if amplitude_invalid:
                    invalid_amplitude_frames += 1

                timestamp = row.get("timestamp")
                if not _finite_number(timestamp):
                    structural_errors.append(f"invalid_timestamp_line_{line_number}")
                else:
                    timestamp_float = float(timestamp)
                    timestamps.append(timestamp_float)
                    if previous_timestamp is not None and timestamp_float < previous_timestamp:
                        timestamp_inversions += 1
                    previous_timestamp = timestamp_float

                sequence = _integer(row.get("sequence"))
                if sequence is None or sequence < 0:
                    structural_errors.append(f"invalid_sequence_line_{line_number}")
                else:
                    sequences.append(sequence)
                    if sequence in seen_sequences:
                        duplicate_sequences += 1
                    else:
                        seen_sequences.add(sequence)
                        if previous_forward_sequence is not None:
                            delta = (sequence - previous_forward_sequence) % (1 << 32)
                            if 0 < delta < (1 << 31):
                                sequence_gap_count += delta - 1
                            else:
                                sequence_reversals += 1
                        previous_forward_sequence = sequence

                rssi = row.get("rssi_dbm")
                if _finite_number(rssi):
                    rssi_values.append(float(rssi))
                else:
                    invalid_rssi_frames += 1

    fail_reasons: list[str] = []
    if parse_errors:
        fail_reasons.append(f"jsonl_parse_errors:{len(parse_errors)}")
    if structural_errors:
        fail_reasons.append(f"invalid_frame_structure:{len(structural_errors)}")
    if metadata_count != 1:
        fail_reasons.append(f"session_metadata_count:{metadata_count}")
    if frame_count == 0:
        fail_reasons.append("frame_count_zero")
    if len(labels) != 1 or next(iter(labels), "") not in ALLOWED_LABELS:
        fail_reasons.append(f"invalid_or_mixed_label:{_display_values(sorted(labels))}")
    path_label = path.parent.name.upper()
    if path_label in ALLOWED_LABELS and labels != {path_label}:
        fail_reasons.append(
            f"label_directory_mismatch:{_display_values(sorted(labels))}!={path_label}"
        )
    if len(session_ids) != 1:
        fail_reasons.append(f"missing_or_mixed_session_id:{_display_values(sorted(session_ids))}")
    if sources != {"esp32"}:
        fail_reasons.append(f"source_not_esp32:{_display_values(sorted(sources))}")
    if subcarrier_counts != {rules.expected_subcarrier_count}:
        fail_reasons.append(
            f"subcarrier_count_not_{rules.expected_subcarrier_count}:"
            f"{_display_values(sorted(subcarrier_counts))}"
        )
    if amplitude_counts != {rules.expected_amplitude_count}:
        fail_reasons.append(
            f"amplitude_count_not_{rules.expected_amplitude_count}:"
            f"{_display_values(sorted(amplitude_counts))}"
        )
    if amplitude_lengths != {rules.expected_amplitude_length}:
        fail_reasons.append(
            f"amplitude_length_not_{rules.expected_amplitude_length}:"
            f"{_display_values(sorted(amplitude_lengths))}"
        )
    if invalid_amplitude_frames:
        fail_reasons.append(f"invalid_amplitude_frames:{invalid_amplitude_frames}")
    if timestamp_inversions:
        fail_reasons.append(f"timestamp_inversions:{timestamp_inversions}")

    duration_sec: float | None = None
    estimated_fps: float | None = None
    if len(timestamps) >= 2:
        duration_sec = timestamps[-1] - timestamps[0]
        if duration_sec > 0:
            estimated_fps = (len(timestamps) - 1) / duration_sec
    elif len(timestamps) == 1:
        duration_sec = 0.0

    if duration_sec is None or duration_sec < rules.fail_min_duration_sec:
        fail_reasons.append(
            "duration_below_collection_minimum:"
            f"{duration_sec if duration_sec is not None else 'unavailable'}"
        )
    elif not rules.duration_warn_min_sec <= duration_sec <= rules.duration_warn_max_sec:
        warnings.append(f"duration_outside_30s_target:{duration_sec:.3f}")

    if estimated_fps is None:
        if frame_count > 0:
            warnings.append("fps_unavailable")
    elif not rules.fps_warn_min <= estimated_fps <= rules.fps_warn_max:
        warnings.append(f"fps_outside_usual_range:{estimated_fps:.3f}")

    denominator = frame_count + sequence_gap_count
    sequence_gap_rate = sequence_gap_count / denominator if denominator else 0.0
    if sequence_gap_rate > rules.sequence_gap_strong_warn_rate:
        warnings.append(f"sequence_gap_rate_strong_warning:{sequence_gap_rate:.6f}")
    elif sequence_gap_rate > rules.sequence_gap_warn_rate:
        warnings.append(f"sequence_gap_rate_warning:{sequence_gap_rate:.6f}")
    if duplicate_sequences:
        warnings.append(f"duplicate_sequences:{duplicate_sequences}")
    if sequence_reversals:
        warnings.append(f"sequence_reversals:{sequence_reversals}")
    if len(node_ids) != 1:
        warnings.append(f"node_id_count:{len(node_ids)}")
    elif node_ids != {rules.expected_node_id}:
        warnings.append(
            f"unexpected_node_id:{_display_values(sorted(node_ids))};"
            f"expected={rules.expected_node_id}"
        )
    if invalid_rssi_frames:
        warnings.append(f"invalid_rssi_frames:{invalid_rssi_frames}")
    if rssi_values and max(rssi_values) - min(rssi_values) > rules.rssi_range_warn_db:
        warnings.append(
            f"rssi_range_high:{max(rssi_values) - min(rssi_values):.3f}_db"
        )

    label_value = next(iter(labels)) if len(labels) == 1 else path.parent.name.upper()
    session_value = next(iter(session_ids)) if len(session_ids) == 1 else path.stem
    status = "FAIL" if fail_reasons else "WARN" if warnings else "PASS"
    return QcResult(
        file_path=str(path),
        session_id=session_value,
        label=label_value,
        dataset_role=dataset_role(session_value),
        frame_count=frame_count,
        duration_sec=duration_sec,
        estimated_fps=estimated_fps,
        first_sequence=sequences[0] if sequences else None,
        last_sequence=sequences[-1] if sequences else None,
        sequence_gap_count=sequence_gap_count,
        sequence_gap_rate=sequence_gap_rate,
        duplicate_sequences=duplicate_sequences,
        sequence_reversals=sequence_reversals,
        node_ids=tuple(sorted(node_ids)),
        sources=tuple(sorted(sources)),
        mean_rssi=sum(rssi_values) / len(rssi_values) if rssi_values else None,
        min_rssi=min(rssi_values) if rssi_values else None,
        max_rssi=max(rssi_values) if rssi_values else None,
        subcarrier_counts=tuple(sorted(subcarrier_counts)),
        amplitude_counts=tuple(sorted(amplitude_counts)),
        amplitude_lengths=tuple(sorted(amplitude_lengths)),
        invalid_amplitude_frames=invalid_amplitude_frames,
        timestamp_inversions=timestamp_inversions,
        timestamps_monotonic=timestamp_inversions == 0,
        parse_error_count=len(parse_errors),
        fail_reasons=tuple(fail_reasons),
        warn_reasons=tuple(warnings),
        qc_status=status,
        heuristic_name=rules.name,
    )


def scan_recordings(directory: str | Path) -> list[QcResult]:
    """Recursively analyze every JSONL file under a dataset directory."""

    root = Path(directory)
    return [analyze_recording(path) for path in sorted(root.rglob("*.jsonl")) if path.is_file()]


def _manifest_value(value: float | int | None, digits: int = 3) -> str | int:
    if value is None:
        return ""
    return round(value, digits) if isinstance(value, float) else value


def write_manifest(results: Iterable[QcResult], manifest_path: str | Path) -> Path:
    """Write a derived CSV manifest. Source JSONL files remain untouched."""

    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for result in results:
            reasons = (*result.fail_reasons, *result.warn_reasons)
            writer.writerow(
                {
                    "session_id": result.session_id,
                    "label": result.label,
                    "dataset_role": result.dataset_role,
                    "file_path": result.file_path,
                    "frames": result.frame_count,
                    "duration_sec": _manifest_value(result.duration_sec),
                    "fps": _manifest_value(result.estimated_fps),
                    "node_id": _display_values(result.node_ids),
                    "source": _display_values(result.sources),
                    "mean_rssi": _manifest_value(result.mean_rssi),
                    "min_rssi": _manifest_value(result.min_rssi),
                    "max_rssi": _manifest_value(result.max_rssi),
                    "first_sequence": _manifest_value(result.first_sequence),
                    "last_sequence": _manifest_value(result.last_sequence),
                    "sequence_gap_count": result.sequence_gap_count,
                    "sequence_gap_rate": round(result.sequence_gap_rate, 6),
                    "duplicate_sequences": result.duplicate_sequences,
                    "subcarrier_count": _display_values(result.subcarrier_counts),
                    "amplitude_count": _display_values(result.amplitude_counts),
                    "amplitude_lengths": _display_values(result.amplitude_lengths),
                    "invalid_amplitude_frames": result.invalid_amplitude_frames,
                    "timestamps_monotonic": str(result.timestamps_monotonic).lower(),
                    "qc_status": result.qc_status,
                    "qc_reasons": ";".join(reasons),
                }
            )
    return path


def _number(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "-" if value is None else f"{value:.{digits}f}{suffix}"


def format_report(result: QcResult) -> str:
    reasons = (*result.fail_reasons, *result.warn_reasons)
    return "\n".join(
        (
            "## Activity Recording QC",
            "",
            f"Heuristic     : {result.heuristic_name} (collection QC; not a paper/model threshold)",
            f"Session       : {result.session_id}",
            f"Label         : {result.label}",
            f"Dataset role  : {result.dataset_role}",
            f"File          : {result.file_path}",
            f"Frames        : {result.frame_count}",
            f"Duration      : {_number(result.duration_sec, 2, ' s')}",
            f"FPS           : {_number(result.estimated_fps, 2)}",
            f"Source        : {_display_values(result.sources)}",
            f"Node          : {_display_values(result.node_ids)}",
            f"Subcarriers   : {_display_values(result.subcarrier_counts)}",
            f"Amplitude cnt : {_display_values(result.amplitude_counts)}",
            f"Amplitude len : {_display_values(result.amplitude_lengths)}",
            f"RSSI mean/min/max : {_number(result.mean_rssi)} / {_number(result.min_rssi)} / {_number(result.max_rssi)} dBm",
            f"Sequence      : {_display_values((result.first_sequence,) if result.first_sequence is not None else ())} -> {_display_values((result.last_sequence,) if result.last_sequence is not None else ())}",
            f"Sequence gaps : {result.sequence_gap_count} ({result.sequence_gap_rate:.2%})",
            f"Duplicate seq : {result.duplicate_sequences}",
            f"Invalid amplitude frames : {result.invalid_amplitude_frames}",
            f"Timestamp monotonic       : {'yes' if result.timestamps_monotonic else 'no'}",
            "",
            f"Result        : {result.qc_status}",
            f"Reasons       : {'; '.join(reasons) if reasons else '-'}",
        )
    )


def format_summary(results: Sequence[QcResult]) -> str:
    headers = ("Session", "Label", "Role", "Frames", "Sec", "FPS", "Gap%", "RSSI", "QC")
    rows = [
        (
            result.session_id,
            result.label,
            result.dataset_role,
            str(result.frame_count),
            _number(result.duration_sec, 1),
            _number(result.estimated_fps, 1),
            f"{result.sequence_gap_rate * 100:.2f}",
            _number(result.mean_rssi, 1),
            result.qc_status,
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ] if rows else [len(header) for header in headers]

    def render(row: Sequence[str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join(("## Dataset QC Summary", "", render(headers), separator, *(render(row) for row in rows)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only QC for RuView activity CSI JSONL recordings. "
            "PASS/WARN/FAIL thresholds are collection heuristics, not research claims."
        )
    )
    parser.add_argument("path", type=Path, help="one activity JSONL file or activity directory")
    parser.add_argument(
        "--manifest",
        nargs="?",
        const="",
        metavar="CSV_PATH",
        help="for a directory scan, write CSV (default: PATH/dataset_manifest.csv)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="for a directory scan, omit per-recording detail",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    target = args.path
    if not target.exists():
        parser.error(f"path does not exist: {target}")

    if target.is_file():
        if args.manifest is not None:
            parser.error("--manifest is available only for a directory scan")
        result = analyze_recording(target)
        print(format_report(result))
        return 1 if result.qc_status == "FAIL" else 0

    results = scan_recordings(target)
    if not results:
        print(f"No JSONL recordings found under {target}", file=sys.stderr)
        return 1
    if not args.summary_only:
        print("\n\n".join(format_report(result) for result in results))
        print()
    print(format_summary(results))
    if args.manifest is not None:
        manifest_path = Path(args.manifest) if args.manifest else target / "dataset_manifest.csv"
        print(f"\nManifest      : {write_manifest(results, manifest_path)}")
    return 1 if any(result.qc_status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
