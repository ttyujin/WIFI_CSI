import numpy as np
from sklearn.preprocessing import StandardScaler

from validate_heightLayoutA_cv import (
    read_frames,
    build_windows,
)

LABELS = ["SITTING", "MOVING", "LYING"]

TRAIN = [
    ("SITTING", "sitting_heightLayoutB_stable_001"),
    ("SITTING", "sitting_heightLayoutB_stable_002"),
    ("SITTING", "sitting_heightLayoutB_stable_003"),

    ("MOVING", "moving_heightLayoutB_stable_001"),
    ("MOVING", "moving_heightLayoutB_stable_002"),
    ("MOVING", "moving_heightLayoutB_stable_003"),

    ("LYING", "lying_heightLayoutB_stable_001"),
    ("LYING", "lying_heightLayoutB_stable_002"),
    ("LYING", "lying_heightLayoutB_stable_003"),
]

TODAY = [
    ("SITTING", "sitting_heightLayoutB_stable_holdout_20260908"),
    ("SITTING", "sitting_heightLayoutB_stable_repeat_20260908_01"),
    ("SITTING", "sitting_heightLayoutB_stable_repeat_20260908_02"),

    ("MOVING", "moving_heightLayoutB_stable_holdout_20260908"),
    ("MOVING", "moving_heightLayoutB_stable_repeat_20260908_01"),
    ("MOVING", "moving_heightLayoutB_stable_repeat_20260908_02"),

    ("LYING", "lying_heightLayoutB_stable_holdout_20260908"),
    ("LYING", "lying_heightLayoutB_stable_repeat_20260908_01"),
    ("LYING", "lying_heightLayoutB_stable_repeat_20260908_02_retry"),
]


def load_session(label, session_id):
    frames = read_frames(label, session_id)
    windows = build_windows(frames)

    return np.asarray(windows, dtype=np.float64)


def main():
    train_x = []
    train_y = []

    print("=" * 78)
    print("LOAD ORIGINAL TRAINING DATA")
    print("=" * 78)

    for label, session_id in TRAIN:
        x = load_session(label, session_id)

        train_x.extend(x)
        train_y.extend([label] * len(x))

        print(
            f"{session_id:<40} "
            f"label={label:<7} windows={len(x)}"
        )

    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y)

    # 중요:
    # scaler는 기존 9개 TRAIN에만 fit
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)

    centroids = {}

    for label in LABELS:
        centroids[label] = train_scaled[
            train_y == label
        ].mean(axis=0)

    print()
    print("=" * 78)
    print("TODAY vs ORIGINAL TRAINING CENTROIDS")
    print("Lower distance = more similar to old class")
    print("=" * 78)

    for actual_label, session_id in TODAY:
        x = load_session(actual_label, session_id)
        x_scaled = scaler.transform(x)

        distances = {}

        for target_label in LABELS:
            d = np.linalg.norm(
                x_scaled - centroids[target_label],
                axis=1
            )

            distances[target_label] = float(d.mean())

        nearest = min(
            LABELS,
            key=lambda label: distances[label]
        )

        print()
        print("-" * 78)
        print(f"Session : {session_id}")
        print(f"Actual  : {actual_label}")
        print(f"Windows : {len(x)}")

        print(
            f"Distance to old SITTING : "
            f"{distances['SITTING']:.4f}"
        )
        print(
            f"Distance to old MOVING  : "
            f"{distances['MOVING']:.4f}"
        )
        print(
            f"Distance to old LYING   : "
            f"{distances['LYING']:.4f}"
        )

        print(f"Nearest old class       : {nearest}")


if __name__ == "__main__":
    main()
