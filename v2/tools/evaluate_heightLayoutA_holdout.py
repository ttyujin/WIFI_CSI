import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from train_activity_baseline import (
    LABELS,
    build_model,
)

from validate_heightLayoutA_cv import (
    FOLDS,
    read_frames,
    build_windows,
)


# ============================================================
# TRAIN
# heightLayoutA 12개 세션만 사용
# ============================================================

TRAIN_SESSIONS = [
    ("SITTING", "sitting_heightLayoutA_001"),
    ("SITTING", "sitting_heightLayoutA_002"),
    ("SITTING", "sitting_heightLayoutA_003"),
    ("SITTING", "sitting_heightLayoutA_004"),

    ("MOVING", "moving_heightLayoutA_001"),
    ("MOVING", "moving_heightLayoutA_002"),
    ("MOVING", "moving_heightLayoutA_003"),
    ("MOVING", "moving_heightLayoutA_006"),

    ("LYING", "lying_heightLayoutA_001"),
    ("LYING", "lying_heightLayoutA_002"),
    ("LYING", "lying_heightLayoutA_004"),
    ("LYING", "lying_heightLayoutA_005"),
]


# ============================================================
# FINAL HOLDOUT
# 절대 학습에 사용하지 않는다.
# ============================================================

HOLDOUT_SESSIONS = [
    ("SITTING", "sitting_heightLayoutA_holdout_002"),
    ("MOVING", "moving_heightLayoutA_holdout_001"),
    ("LYING", "lying_heightLayoutA_holdout_001"),
]


def main():

    train_ids = {session_id for _, session_id in TRAIN_SESSIONS}
    holdout_ids = {session_id for _, session_id in HOLDOUT_SESSIONS}

    if len(TRAIN_SESSIONS) != 12:
        raise RuntimeError("Training set must contain exactly 12 sessions")

    if len(HOLDOUT_SESSIONS) != 3:
        raise RuntimeError("Holdout must contain exactly 3 sessions")

    if train_ids & holdout_ids:
        raise RuntimeError("TRAIN / HOLDOUT leakage detected")

    # 실패했던 3.82초 파일이 실수로 들어가는 것을 방지
    if "sitting_heightLayoutA_holdout_001" in holdout_ids:
        raise RuntimeError(
            "Failed sitting holdout_001 must not be used"
        )

    print("=" * 76)
    print("TRAINING DATA")
    print("=" * 76)

    x_train = []
    y_train = []

    for label, session_id in TRAIN_SESSIONS:

        frames = read_frames(label, session_id)
        windows = build_windows(frames)

        print(
            f"{session_id:<36} "
            f"label={label:<7} "
            f"frames={len(frames):>4} "
            f"windows={len(windows):>2}"
        )

        for feature in windows:
            x_train.append(feature)
            y_train.append(label)

    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train)

    print()
    print(f"Total training windows : {len(x_train)}")
    print(f"Feature dimension      : {x_train.shape[1]}")

    print()
    print("Training temporary heightLayoutA model...")

    model = build_model("LOGISTIC_REGRESSION")
    model.fit(x_train, y_train)

    print("Training complete.")

    print()
    print("=" * 76)
    print("FINAL HOLDOUT EVALUATION")
    print("HOLDOUT HAS NEVER BEEN USED FOR TRAINING")
    print("=" * 76)

    all_true = []
    all_pred = []

    session_correct = 0

    for actual_label, session_id in HOLDOUT_SESSIONS:

        frames = read_frames(actual_label, session_id)
        windows = build_windows(frames)

        x_test = np.asarray(windows, dtype=np.float64)

        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)

        model_classes = [
            str(value)
            for value in model.classes_
        ]

        mean_probabilities = {
            label: float(
                probabilities[
                    :,
                    model_classes.index(label)
                ].mean()
            )
            for label in LABELS
        }

        session_prediction = max(
            LABELS,
            key=lambda label: mean_probabilities[label]
        )

        if session_prediction == actual_label:
            session_correct += 1

        counts = {
            label: int(np.sum(predictions == label))
            for label in LABELS
        }

        print()
        print("-" * 76)
        print(f"Session : {session_id}")
        print(f"Actual  : {actual_label}")
        print(f"Frames  : {len(frames)}")
        print(f"Windows : {len(windows)}")

        print(
            "Window predictions : "
            + ", ".join(
                f"{label}={counts[label]}"
                for label in LABELS
            )
        )

        print(
            "Mean probability   : "
            + ", ".join(
                f"{label}={mean_probabilities[label]:.6f}"
                for label in LABELS
            )
        )

        print(
            f"Session prediction : {session_prediction} "
            f"({'CORRECT' if session_prediction == actual_label else 'WRONG'})"
        )

        all_true.extend(
            [actual_label] * len(predictions)
        )

        all_pred.extend(
            predictions.tolist()
        )

    accuracy = accuracy_score(
        all_true,
        all_pred,
    )

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            all_true,
            all_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        all_true,
        all_pred,
        labels=LABELS,
    )

    print()
    print("#" * 76)
    print("FINAL UNSEEN HOLDOUT RESULT")
    print("#" * 76)

    print(
        f"Window accuracy  : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro precision  : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Macro recall     : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Macro F1         : "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"Session accuracy : "
        f"{session_correct}/3 "
        f"({session_correct / 3 * 100:.2f}%)"
    )

    print()
    print("Holdout window confusion matrix")
    print(
        "Actual \\ Pred     "
        "SITTING   MOVING   LYING"
    )

    for label, row in zip(LABELS, matrix):

        print(
            f"{label:<17}"
            f"{row[0]:>7}"
            f"{row[1]:>9}"
            f"{row[2]:>8}"
        )


if __name__ == "__main__":
    main()
