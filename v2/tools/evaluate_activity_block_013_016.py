import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

from activity_inference import ActivityV1Predictor

WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.5
EPSILON = 1e-9
LABELS = ("SITTING", "MOVING", "LYING")

SESSIONS = []
for label in LABELS:
    folder = label
    prefix = label.lower()
    for number in range(13, 17):
        SESSIONS.append(
            (
                label,
                f"{prefix}_{number:03d}",
                Path(f"data/recordings/activity/{folder}/{prefix}_{number:03d}.jsonl")
            )
        )


def read_frames(path):
    frames = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("record_type") == "frame":
                frames.append(row)

    frames.sort(key=lambda row: float(row["timestamp"]))
    return frames


def build_windows(frames):
    first_timestamp = float(frames[0]["timestamp"])
    last_timestamp = float(frames[-1]["timestamp"])

    windows = []
    window_index = 0

    while True:
        start = first_timestamp + window_index * HOP_SECONDS
        end = start + WINDOW_SECONDS

        if end > last_timestamp + EPSILON:
            break

        selected = [
            frame
            for frame in frames
            if start <= float(frame["timestamp"]) < end
        ]

        if selected:
            windows.append(selected)

        window_index += 1

    return windows


def main():
    predictor = ActivityV1Predictor.from_artifact_dir("models/activity_v1")

    window_confusion = defaultdict(Counter)

    total_windows = 0
    correct_windows = 0
    total_sessions = 0
    correct_sessions = 0

    print("\nCURRENT BLOCK 013-016 — EXISTING activity-v1 ONLY")
    print("No retraining / 3.0s window / 1.5s hop\n")

    for actual_label, session_id, path in SESSIONS:
        frames = read_frames(path)
        windows = build_windows(frames)

        results = [predictor.predict(window) for window in windows]

        counts = Counter(result["activity"] for result in results)

        mean_probabilities = {
            label: float(np.mean([
                result["probabilities"][label]
                for result in results
            ]))
            for label in LABELS
        }

        session_prediction = max(
            mean_probabilities,
            key=mean_probabilities.get
        )

        for result in results:
            predicted = result["activity"]
            window_confusion[actual_label][predicted] += 1

            total_windows += 1
            if predicted == actual_label:
                correct_windows += 1

        total_sessions += 1
        if session_prediction == actual_label:
            correct_sessions += 1

        print("=" * 70)
        print(f"Session            : {session_id}")
        print(f"Actual             : {actual_label}")
        print(f"Frames             : {len(frames)}")
        print(f"Windows            : {len(windows)}")

        print("Window predictions : ", end="")
        print(
            ", ".join(
                f"{label}={counts.get(label, 0)}"
                for label in LABELS
            )
        )

        print(
            "Mean probabilities : "
            + ", ".join(
                f"{label}={mean_probabilities[label]:.6f}"
                for label in LABELS
            )
        )

        print(f"Session prediction : {session_prediction}")
        print(f"Correct            : {session_prediction == actual_label}")

    print("\n" + "#" * 70)
    print("AGGREGATE DIAGNOSTIC")
    print("#" * 70)

    print(
        f"Window accuracy  : "
        f"{correct_windows}/{total_windows} "
        f"({correct_windows / total_windows * 100:.2f}%)"
    )

    print(
        f"Session accuracy : "
        f"{correct_sessions}/{total_sessions} "
        f"({correct_sessions / total_sessions * 100:.2f}%)"
    )

    print("\nWindow confusion matrix")
    print("Actual \\ Pred     SITTING   MOVING   LYING")

    for actual in LABELS:
        print(
            f"{actual:<17}"
            f"{window_confusion[actual]['SITTING']:>7}"
            f"{window_confusion[actual]['MOVING']:>9}"
            f"{window_confusion[actual]['LYING']:>8}"
        )


if __name__ == "__main__":
    main()
