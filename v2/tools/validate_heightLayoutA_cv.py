import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from train_activity_baseline import (
    LABELS,
    build_model,
    extract_normalized_features,
)

WINDOW_SECONDS = 3.0
HOP_SECONDS = 1.5
EPSILON = 1e-9
EXPECTED_SUBCARRIERS = 306

# 정확히 이 12세션만 사용한다.
FOLDS = [
    {
        "SITTING": "sitting_heightLayoutA_001",
        "MOVING": "moving_heightLayoutA_001",
        "LYING": "lying_heightLayoutA_001",
    },
    {
        "SITTING": "sitting_heightLayoutA_002",
        "MOVING": "moving_heightLayoutA_002",
        "LYING": "lying_heightLayoutA_002",
    },
    {
        "SITTING": "sitting_heightLayoutA_003",
        "MOVING": "moving_heightLayoutA_003",
        "LYING": "lying_heightLayoutA_004",
    },
    {
        "SITTING": "sitting_heightLayoutA_004",
        "MOVING": "moving_heightLayoutA_006",
        "LYING": "lying_heightLayoutA_005",
    },
]


def session_path(label, session_id):
    return Path(
        f"data/recordings/activity/{label}/{session_id}.jsonl"
    )


def read_frames(label, session_id):
    path = session_path(label, session_id)

    if not path.exists():
        raise FileNotFoundError(path)

    frames = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            if row.get("record_type") != "frame":
                continue

            amplitude = row.get("amplitude")

            if not isinstance(amplitude, list):
                raise ValueError(
                    f"{session_id} line {line_number}: amplitude missing"
                )

            if len(amplitude) != EXPECTED_SUBCARRIERS:
                raise ValueError(
                    f"{session_id} line {line_number}: "
                    f"amplitude length={len(amplitude)}"
                )

            if int(row.get("subcarrier_count", -1)) != EXPECTED_SUBCARRIERS:
                raise ValueError(
                    f"{session_id} line {line_number}: "
                    "subcarrier_count != 306"
                )

            if not np.all(np.isfinite(np.asarray(amplitude, dtype=np.float64))):
                raise ValueError(
                    f"{session_id} line {line_number}: non-finite amplitude"
                )

            frames.append(row)

    frames.sort(key=lambda row: float(row["timestamp"]))

    if len(frames) < 2:
        raise RuntimeError(f"Not enough frames: {session_id}")

    return frames


def build_windows(frames):
    first_timestamp = float(frames[0]["timestamp"])
    last_timestamp = float(frames[-1]["timestamp"])

    windows = []
    window_index = 0

    while True:
        start = first_timestamp + window_index * HOP_SECONDS
        end = start + WINDOW_SECONDS

        # 기존 preprocessing과 동일하게 full window만 사용
        if end > last_timestamp + EPSILON:
            break

        selected = [
            row
            for row in frames
            if start <= float(row["timestamp"]) < end
        ]

        if selected:
            amplitude = np.asarray(
                [row["amplitude"] for row in selected],
                dtype=np.float64,
            )

            feature, zero_mean_count = extract_normalized_features(amplitude)

            if feature.shape != (918,):
                raise ValueError(
                    f"Unexpected feature shape: {feature.shape}"
                )

            if not np.all(np.isfinite(feature)):
                raise ValueError("Non-finite feature")

            windows.append(feature)

        window_index += 1

    return windows


def load_dataset():
    sessions = {}

    expected_session_ids = {
        session_id
        for fold in FOLDS
        for session_id in fold.values()
    }

    if len(expected_session_ids) != 12:
        raise RuntimeError(
            f"Expected exactly 12 unique sessions, got {len(expected_session_ids)}"
        )

    if any("holdout" in sid.lower() for sid in expected_session_ids):
        raise RuntimeError("Holdout session accidentally included")

    for fold in FOLDS:
        for label in LABELS:
            session_id = fold[label]

            if session_id in sessions:
                continue

            frames = read_frames(label, session_id)
            windows = build_windows(frames)

            sessions[session_id] = {
                "label": label,
                "frames": len(frames),
                "windows": windows,
            }

            print(
                f"Loaded {session_id:<32} "
                f"label={label:<7} "
                f"frames={len(frames):>4} "
                f"windows={len(windows):>2}"
            )

    return sessions


def main():
    sessions = load_dataset()

    all_true = []
    all_pred = []

    fold_accuracies = []
    fold_macro_f1 = []

    total_session_correct = 0
    total_session_count = 0

    print()
    print("=" * 76)
    print("heightLayoutA 4-FOLD SESSION-LEVEL CROSS VALIDATION")
    print("NORMALIZED 918 features")
    print("StandardScaler + LogisticRegression")
    print("3.0 s window / 1.5 s hop")
    print("NO HOLDOUT / NO SESSION LEAKAGE")
    print("=" * 76)

    for fold_index, test_fold in enumerate(FOLDS, start=1):

        test_session_ids = set(test_fold.values())

        x_train = []
        y_train = []

        for session_id, info in sessions.items():
            if session_id in test_session_ids:
                continue

            for feature in info["windows"]:
                x_train.append(feature)
                y_train.append(info["label"])

        x_train = np.asarray(x_train, dtype=np.float64)

        model = build_model("LOGISTIC_REGRESSION")
        model.fit(x_train, y_train)

        fold_true = []
        fold_pred = []

        print()
        print("-" * 76)
        print(f"FOLD {fold_index}")
        print("-" * 76)

        fold_session_correct = 0

        for actual_label in LABELS:
            session_id = test_fold[actual_label]
            session_windows = np.asarray(
                sessions[session_id]["windows"],
                dtype=np.float64,
            )

            predictions = model.predict(session_windows)
            probabilities = model.predict_proba(session_windows)

            model_classes = [str(value) for value in model.classes_]

            mean_probabilities = {
                label: float(
                    probabilities[:, model_classes.index(label)].mean()
                )
                for label in LABELS
            }

            session_prediction = max(
                LABELS,
                key=lambda label: mean_probabilities[label]
            )

            correct = session_prediction == actual_label

            if correct:
                fold_session_correct += 1
                total_session_correct += 1

            total_session_count += 1

            fold_true.extend(
                [actual_label] * len(predictions)
            )
            fold_pred.extend(predictions.tolist())

            all_true.extend(
                [actual_label] * len(predictions)
            )
            all_pred.extend(predictions.tolist())

            counts = {
                label: int(np.sum(predictions == label))
                for label in LABELS
            }

            print(f"Session : {session_id}")
            print(f"Actual  : {actual_label}")
            print(
                "Windows : "
                + ", ".join(
                    f"{label}={counts[label]}"
                    for label in LABELS
                )
            )
            print(
                "Mean P  : "
                + ", ".join(
                    f"{label}={mean_probabilities[label]:.6f}"
                    for label in LABELS
                )
            )
            print(
                f"Predict : {session_prediction} "
                f"({'CORRECT' if correct else 'WRONG'})"
            )
            print()

        fold_accuracy = accuracy_score(
            fold_true,
            fold_pred,
        )

        _, _, macro_f1, _ = precision_recall_fscore_support(
            fold_true,
            fold_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        fold_accuracies.append(fold_accuracy)
        fold_macro_f1.append(macro_f1)

        print(
            f"Fold window accuracy : {fold_accuracy * 100:.2f}%"
        )
        print(
            f"Fold macro F1        : {macro_f1 * 100:.2f}%"
        )
        print(
            f"Fold session accuracy: "
            f"{fold_session_correct}/3 "
            f"({fold_session_correct / 3 * 100:.2f}%)"
        )

    overall_accuracy = accuracy_score(
        all_true,
        all_pred,
    )

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        all_true,
        all_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_true,
        all_pred,
        labels=LABELS,
    )

    print()
    print("#" * 76)
    print("FINAL HEIGHTLAYOUTA CV RESULT")
    print("#" * 76)

    print(
        f"Fold window accuracy : "
        f"{np.mean(fold_accuracies) * 100:.2f} "
        f"± {np.std(fold_accuracies) * 100:.2f}%"
    )

    print(
        f"Fold macro F1        : "
        f"{np.mean(fold_macro_f1) * 100:.2f} "
        f"± {np.std(fold_macro_f1) * 100:.2f}%"
    )

    print(
        f"OOF window accuracy  : "
        f"{overall_accuracy * 100:.2f}%"
    )

    print(
        f"OOF macro precision  : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"OOF macro recall     : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"OOF macro F1         : "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"Session accuracy     : "
        f"{total_session_correct}/{total_session_count} "
        f"({total_session_correct / total_session_count * 100:.2f}%)"
    )

    print()
    print("OOF window confusion matrix")
    print("Actual \\ Pred     SITTING   MOVING   LYING")

    for label, row in zip(LABELS, matrix):
        print(
            f"{label:<17}"
            f"{row[0]:>7}"
            f"{row[1]:>9}"
            f"{row[2]:>8}"
        )


if __name__ == "__main__":
    main()
