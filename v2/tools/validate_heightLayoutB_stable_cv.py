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

FOLDS = [
    {
        "SITTING": "sitting_heightLayoutB_stable_001",
        "MOVING":  "moving_heightLayoutB_stable_001",
        "LYING":   "lying_heightLayoutB_stable_001",
    },
    {
        "SITTING": "sitting_heightLayoutB_stable_002",
        "MOVING":  "moving_heightLayoutB_stable_002",
        "LYING":   "lying_heightLayoutB_stable_002",
    },
    {
        "SITTING": "sitting_heightLayoutB_stable_003",
        "MOVING":  "moving_heightLayoutB_stable_003",
        "LYING":   "lying_heightLayoutB_stable_003",
    },
]


def main():
    sessions = {}

    for fold in FOLDS:
        for label in LABELS:
            session_id = fold[label]

            frames = read_frames(label, session_id)
            windows = build_windows(frames)

            sessions[session_id] = {
                "label": label,
                "windows": windows,
            }

            print(
                f"Loaded {session_id:<38} "
                f"label={label:<7} "
                f"frames={len(frames):>4} "
                f"windows={len(windows):>2}"
            )

    all_true = []
    all_pred = []

    session_correct = 0

    print()
    print("=" * 76)
    print("heightLayoutB_stable — 3-FOLD SESSION-LEVEL CV")
    print("NORMALIZED 918 features")
    print("StandardScaler + LogisticRegression")
    print("NO SESSION LEAKAGE")
    print("=" * 76)

    for fold_index, test_fold in enumerate(FOLDS, start=1):

        test_ids = set(test_fold.values())

        x_train = []
        y_train = []

        for session_id, info in sessions.items():
            if session_id in test_ids:
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

        for actual_label in LABELS:
            session_id = test_fold[actual_label]

            x_test = np.asarray(
                sessions[session_id]["windows"],
                dtype=np.float64,
            )

            predictions = model.predict(x_test)
            probabilities = model.predict_proba(x_test)

            model_classes = [str(c) for c in model.classes_]

            mean_probs = {
                label: float(
                    probabilities[:, model_classes.index(label)].mean()
                )
                for label in LABELS
            }

            session_prediction = max(
                LABELS,
                key=lambda label: mean_probs[label],
            )

            if session_prediction == actual_label:
                session_correct += 1

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
                    f"{label}={mean_probs[label]:.6f}"
                    for label in LABELS
                )
            )
            print(
                f"Predict : {session_prediction} "
                f"({'CORRECT' if session_prediction == actual_label else 'WRONG'})"
            )
            print()

            fold_true.extend(
                [actual_label] * len(predictions)
            )
            fold_pred.extend(
                predictions.tolist()
            )

            all_true.extend(
                [actual_label] * len(predictions)
            )
            all_pred.extend(
                predictions.tolist()
            )

        fold_acc = accuracy_score(
            fold_true,
            fold_pred,
        )

        _, _, fold_f1, _ = precision_recall_fscore_support(
            fold_true,
            fold_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        print(
            f"Fold window accuracy : {fold_acc * 100:.2f}%"
        )
        print(
            f"Fold macro F1        : {fold_f1 * 100:.2f}%"
        )

    overall_acc = accuracy_score(
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
    print("FINAL HEIGHTLAYOUTB_STABLE CV RESULT")
    print("#" * 76)

    print(
        f"OOF window accuracy  : {overall_acc * 100:.2f}%"
    )
    print(
        f"OOF macro precision  : {precision * 100:.2f}%"
    )
    print(
        f"OOF macro recall     : {recall * 100:.2f}%"
    )
    print(
        f"OOF macro F1         : {macro_f1 * 100:.2f}%"
    )
    print(
        f"Session accuracy     : "
        f"{session_correct}/9 "
        f"({session_correct / 9 * 100:.2f}%)"
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
