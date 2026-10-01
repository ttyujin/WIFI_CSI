import json
from pathlib import Path
from collections import Counter

import numpy as np

from activity_inference import ActivityV1Predictor

WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.5
EPSILON = 1e-9

SESSIONS = [
    ("SITTING", Path("data/recordings/activity/SITTING/sitting_heightPilot_001.jsonl")),
    ("MOVING",  Path("data/recordings/activity/MOVING/moving_heightPilot_001.jsonl")),
    ("LYING",   Path("data/recordings/activity/LYING/lying_heightPilot_001.jsonl")),
]


def read_frames(path):
    frames = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("record_type") != "frame":
                continue

            frames.append(row)

    frames.sort(key=lambda x: float(x["timestamp"]))
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

    for actual_label, path in SESSIONS:
        frames = read_frames(path)
        windows = build_windows(frames)

        results = [predictor.predict(window) for window in windows]

        counts = Counter(result["activity"] for result in results)

        mean_probabilities = {
            label: float(np.mean([
                result["probabilities"][label]
                for result in results
            ]))
            for label in ("SITTING", "MOVING", "LYING")
        }

        session_prediction = max(
            mean_probabilities,
            key=mean_probabilities.get
        )

        print("=" * 60)
        print(f"Actual label       : {actual_label}")
        print(f"File               : {path}")
        print(f"Frames             : {len(frames)}")
        print(f"3s / 1.5s windows  : {len(windows)}")
        print()
        print("Window predictions :")
        for label in ("SITTING", "MOVING", "LYING"):
            count = counts.get(label, 0)
            pct = count / len(windows) * 100 if windows else 0
            print(f"  {label:<7}: {count:>3} ({pct:6.2f}%)")

        print()
        print("Mean probabilities :")
        for label in ("SITTING", "MOVING", "LYING"):
            print(f"  {label:<7}: {mean_probabilities[label]:.6f}")

        print()
        print(f"Session prediction : {session_prediction}")
        print(f"Correct            : {session_prediction == actual_label}")
        print()


if __name__ == "__main__":
    main()
