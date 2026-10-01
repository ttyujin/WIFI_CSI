import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from train_activity_baseline import LABELS, build_model
from validate_heightLayoutA_cv import read_frames, build_windows


TODAY_TRAIN = [
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

OLD_TEST = [
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


def load_windows(label, session_id):
    frames = read_frames(label, session_id)
    return np.asarray(build_windows(frames), dtype=np.float64)


def main():

    x_train = []
    y_train = []

    print("=" * 76)
    print("TRAIN ON 2026-09-08 DATA")
    print("=" * 76)

    for label, sid in TODAY_TRAIN:
        x = load_windows(label, sid)

        print(
            f"{sid:<50} "
            f"label={label:<7} windows={len(x)}"
        )

        x_train.extend(x)
        y_train.extend([label] * len(x))

    x_train = np.asarray(x_train, dtype=np.float64)

    model = build_model("LOGISTIC_REGRESSION")
    model.fit(x_train, y_train)

    all_true = []
    all_pred = []

    session_correct = 0

    print()
    print("=" * 76)
    print("TEST ON ORIGINAL heightLayoutB_stable DATA")
    print("=" * 76)

    for actual_label, sid in OLD_TEST:

        x = load_windows(actual_label, sid)

        pred = model.predict(x)
        proba = model.predict_proba(x)

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
        print(f"Session : {sid}")
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
                f"{label}={mean_probs[label]:.6f}"
                for label in LABELS
            )
        )

        print(
            f"Predict : {session_pred} "
            f"({'CORRECT' if session_pred == actual_label else 'WRONG'})"
        )

        all_true.extend([actual_label] * len(pred))
        all_pred.extend(pred.tolist())

    accuracy = accuracy_score(all_true, all_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
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
    print("REVERSE TIME-BLOCK RESULT")
    print("#" * 76)

    print(f"Window accuracy  : {accuracy * 100:.2f}%")
    print(f"Macro precision  : {precision * 100:.2f}%")
    print(f"Macro recall     : {recall * 100:.2f}%")
    print(f"Macro F1         : {f1 * 100:.2f}%")

    print(
        f"Session accuracy : "
        f"{session_correct}/9 "
        f"({session_correct / 9 * 100:.2f}%)"
    )

    print()
    print("Confusion matrix")
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
