import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from validate_heightLayoutA_cv import read_frames, build_windows

SESSIONS = [
    ("SITTING", "sitting_heightLayoutB_stable_001"),
    ("MOVING",  "moving_heightLayoutB_stable_001"),
    ("LYING",   "lying_heightLayoutB_stable_001"),
]

LABELS = ["SITTING", "MOVING", "LYING"]

X = []
y = []

for label, session_id in SESSIONS:
    frames = read_frames(label, session_id)
    windows = build_windows(frames)

    print(
        f"{session_id}: "
        f"frames={len(frames)}, windows={len(windows)}"
    )

    X.extend(windows)
    y.extend([label] * len(windows))

X = np.asarray(X, dtype=np.float64)
y = np.asarray(y)

Xs = StandardScaler().fit_transform(X)

centroids = {}
dispersion = {}

for label in LABELS:
    Xi = Xs[y == label]
    centroids[label] = Xi.mean(axis=0)

    dispersion[label] = np.linalg.norm(
        Xi - centroids[label],
        axis=1
    ).mean()

print("\n=== PAIRWISE SEPARATION ===")

for i in range(len(LABELS)):
    for j in range(i + 1, len(LABELS)):
        a = LABELS[i]
        b = LABELS[j]

        distance = np.linalg.norm(
            centroids[a] - centroids[b]
        )

        within = (
            dispersion[a] + dispersion[b]
        ) / 2

        ratio = distance / within

        print(
            f"{a} vs {b}: "
            f"distance={distance:.4f}, "
            f"ratio={ratio:.4f}"
        )

print(
    "\nSilhouette score:",
    round(silhouette_score(Xs, y), 4)
)

print("\nPILOT ONLY - NOT CLASSIFICATION ACCURACY")
