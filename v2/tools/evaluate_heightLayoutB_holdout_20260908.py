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
    read_frames,
    build_windows,
)

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

HOLDOUT = [
    ("SITTING", "sitting_heightLayoutB_stable_holdout_20260908"),
    ("MOVING", "moving_heightLayoutB_stable_holdout_20260908"),
    ("LYING", "lying_heightLayoutB_stable_holdout_20260908"),
]


def main():
    train_ids = {sid for _, sid in TRAIN}
    holdout_ids = {sid for _, sid in HOLDOUT}

    if train_ids & holdout_ids:
        raise RuntimeError("TRAIN/HOLDOUT leakage detected")

    print("=" * 76)
    print("TRAIN — heightLayoutB_stable 9 sessions")
    print("=" * 76)

    x_train = []
    y_train = []

    for label, session_id in TRAIN:
        frames = read_frames(label, session_id)
        windows = build_windows(frames)

        print(
            f"{session_id:<40} "
            f"frames={len(frames):>4} "
            f"windows={len(windows):>2}"
        )

        for feature in windows:
            x_train.append(feature)
            y_train.append(label)

    x_train = np.asarray(x_train, dtype=np.float64)

    print()
    print(f"Training windows : {len(x_train)}")
    print(f"Feature dim      : {x_train.shape[1]}")

    model = build_model("LOGISTIC_REGRESSION")
    model.fit(x_train, y_train)

    print()
    print("=" * 76)
    print("TIME-SEPARATED HOLDOUT — 2026-09-08")
    print("HOLDOUT WAS NOT USED FOR TRAINING")
    print("=" * 76)

    all_true = []
    all_pred = []
    session_correct = 0

    for actual_label, session_id in HOLDOUT:
        frames = read_frames(actual_label, session_id)
        windows = build_windows(frames)

        x_test = np.asarray(windows, dtype=np.float64)

        pred = model.predict(x_test)
        proba = model.predict_proba(x_test)

        classes = [str(c) for c in model.classes_]

        mean_probs = {
            label: float(
                proba[:, classes.index(label)].mean()
            )
            for label in LABELS
        }

        session_pred = max(
            LABELS,
            key=lambda label: mean_probs[label],
        )

        if session_pred == actual_label:
            session_correct += 1

        counts = {
            label: int(np.sum(pred == label))
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
                f"{label}={mean_probs[label]:.6f}"
                for label in LABELS
            )
        )
        print(
            f"Session prediction : {session_pred} "
            f"({'CORRECT' if session_pred == actual_label else 'WRONG'})"
        )

        all_true.extend([actual_label] * len(pred))
        all_pred.extend(pred.tolist())

    acc = accuracy_score(all_true, all_pred)

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
    print("FINAL TIME-SEPARATED HOLDOUT RESULT")
    print("#" * 76)

    print(f"Window accuracy  : {acc * 100:.2f}%")
    print(f"Macro precision  : {precision * 100:.2f}%")
    print(f"Macro recall     : {recall * 100:.2f}%")
    print(f"Macro F1         : {macro_f1 * 100:.2f}%")
    print(
        f"Session accuracy : {session_correct}/3 "
        f"({session_correct / 3 * 100:.2f}%)"
    )

    print()
    print("Holdout confusion matrix")
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
