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
    [
        ("OLD", "SITTING", "sitting_heightLayoutB_stable_001"),
        ("OLD", "MOVING",  "moving_heightLayoutB_stable_001"),
        ("OLD", "LYING",   "lying_heightLayoutB_stable_001"),

        ("TODAY", "SITTING", "sitting_heightLayoutB_stable_holdout_20260908"),
        ("TODAY", "MOVING",  "moving_heightLayoutB_stable_holdout_20260908"),
        ("TODAY", "LYING",   "lying_heightLayoutB_stable_holdout_20260908"),
    ],

    [
        ("OLD", "SITTING", "sitting_heightLayoutB_stable_002"),
        ("OLD", "MOVING",  "moving_heightLayoutB_stable_002"),
        ("OLD", "LYING",   "lying_heightLayoutB_stable_002"),

        ("TODAY", "SITTING", "sitting_heightLayoutB_stable_repeat_20260908_01"),
        ("TODAY", "MOVING",  "moving_heightLayoutB_stable_repeat_20260908_01"),
        ("TODAY", "LYING",   "lying_heightLayoutB_stable_repeat_20260908_01"),
    ],

    [
        ("OLD", "SITTING", "sitting_heightLayoutB_stable_003"),
        ("OLD", "MOVING",  "moving_heightLayoutB_stable_003"),
        ("OLD", "LYING",   "lying_heightLayoutB_stable_003"),

        ("TODAY", "SITTING", "sitting_heightLayoutB_stable_repeat_20260908_02"),
        ("TODAY", "MOVING",  "moving_heightLayoutB_stable_repeat_20260908_02"),
        ("TODAY", "LYING",   "lying_heightLayoutB_stable_repeat_20260908_02_retry"),
    ],
]


MODEL_NAMES = [
    "LOGISTIC_REGRESSION",
    "RANDOM_FOREST",
]


def load_sessions():
    sessions = {}

    all_specs = [
        spec
        for fold in FOLDS
        for spec in fold
    ]

    forbidden = "lying_heightLayoutB_stable_repeat_20260908_02"

    for block, label, sid in all_specs:

        if sid == forbidden:
            raise RuntimeError(
                "Failed mislabeled 3.84 sec file must not be used"
            )

        if sid in sessions:
            continue

        frames = read_frames(label, sid)
        windows = build_windows(frames)

        sessions[sid] = {
            "block": block,
            "label": label,
            "windows": np.asarray(
                windows,
                dtype=np.float64,
            ),
        }

        print(
            f"Loaded {block:<5} "
            f"{sid:<52} "
            f"{label:<7} "
            f"windows={len(windows)}"
        )

    if len(sessions) != 18:
        raise RuntimeError(
            f"Expected 18 sessions, got {len(sessions)}"
        )

    return sessions


def evaluate_model(model_name, sessions):

    all_true = []
    all_pred = []
    all_blocks = []

    session_correct = 0
    session_total = 0

    fold_accuracies = []
    fold_f1s = []

    print()
    print("=" * 86)
    print(f"MODEL: {model_name}")
    print("=" * 86)

    for fold_index, test_specs in enumerate(FOLDS, start=1):

        test_ids = {
            sid
            for _, _, sid in test_specs
        }

        x_train = []
        y_train = []

        for sid, info in sessions.items():

            if sid in test_ids:
                continue

            for feature in info["windows"]:
                x_train.append(feature)
                y_train.append(info["label"])

        x_train = np.asarray(
            x_train,
            dtype=np.float64,
        )

        model = build_model(model_name)
        model.fit(x_train, y_train)

        fold_true = []
        fold_pred = []

        print()
        print("-" * 86)
        print(f"FOLD {fold_index}")
        print("-" * 86)

        for block, actual_label, sid in test_specs:

            x_test = sessions[sid]["windows"]

            pred = model.predict(x_test)
            proba = model.predict_proba(x_test)

            classes = [
                str(c)
                for c in model.classes_
            ]

            mean_probs = {
                label: float(
                    proba[
                        :,
                        classes.index(label)
                    ].mean()
                )
                for label in LABELS
            }

            session_pred = max(
                LABELS,
                key=lambda label: mean_probs[label],
            )

            correct = (
                session_pred == actual_label
            )

            if correct:
                session_correct += 1

            session_total += 1

            counts = {
                label: int(
                    np.sum(pred == label)
                )
                for label in LABELS
            }

            print()
            print(f"[{block}] {sid}")
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
                f"({'CORRECT' if correct else 'WRONG'})"
            )

            fold_true.extend(
                [actual_label] * len(pred)
            )
            fold_pred.extend(
                pred.tolist()
            )

            all_true.extend(
                [actual_label] * len(pred)
            )
            all_pred.extend(
                pred.tolist()
            )

            all_blocks.extend(
                [block] * len(pred)
            )

        fold_acc = accuracy_score(
            fold_true,
            fold_pred,
        )

        _, _, fold_f1, _ = (
            precision_recall_fscore_support(
                fold_true,
                fold_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        )

        fold_accuracies.append(fold_acc)
        fold_f1s.append(fold_f1)

        print()
        print(
            f"Fold accuracy : {fold_acc * 100:.2f}%"
        )
        print(
            f"Fold macro F1 : {fold_f1 * 100:.2f}%"
        )

    overall_acc = accuracy_score(
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
    print("#" * 86)
    print(f"FINAL RESULT — {model_name}")
    print("#" * 86)

    print(
        f"Fold accuracy mean : "
        f"{np.mean(fold_accuracies) * 100:.2f} "
        f"± {np.std(fold_accuracies) * 100:.2f}%"
    )

    print(
        f"Fold macro F1 mean : "
        f"{np.mean(fold_f1s) * 100:.2f} "
        f"± {np.std(fold_f1s) * 100:.2f}%"
    )

    print(
        f"OOF window accuracy : "
        f"{overall_acc * 100:.2f}%"
    )

    print(
        f"OOF macro precision : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"OOF macro recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"OOF macro F1        : "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"Session accuracy    : "
        f"{session_correct}/{session_total} "
        f"({session_correct/session_total*100:.2f}%)"
    )

    print()
    print("Confusion matrix")
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

    # block별 성능
    true_np = np.asarray(all_true)
    pred_np = np.asarray(all_pred)
    block_np = np.asarray(all_blocks)

    block_results = {}

    for block in ["OLD", "TODAY"]:

        mask = (
            block_np == block
        )

        block_acc = accuracy_score(
            true_np[mask],
            pred_np[mask],
        )

        _, _, block_f1, _ = (
            precision_recall_fscore_support(
                true_np[mask],
                pred_np[mask],
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        )

        block_results[block] = (
            block_acc,
            block_f1,
        )

        print()
        print(
            f"{block} accuracy : "
            f"{block_acc * 100:.2f}%"
        )

        print(
            f"{block} macro F1 : "
            f"{block_f1 * 100:.2f}%"
        )

    return {
        "model": model_name,
        "accuracy": overall_acc,
        "macro_f1": macro_f1,
        "session_accuracy":
            session_correct / session_total,
        "old_accuracy":
            block_results["OLD"][0],
        "today_accuracy":
            block_results["TODAY"][0],
        "matrix": matrix,
    }


def main():

    sessions = load_sessions()

    results = []

    for model_name in MODEL_NAMES:
        result = evaluate_model(
            model_name,
            sessions,
        )
        results.append(result)

    print()
    print("=" * 86)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 86)

    print(
        f"{'MODEL':<24}"
        f"{'WINDOW ACC':>12}"
        f"{'MACRO F1':>12}"
        f"{'SESSION ACC':>14}"
        f"{'OLD ACC':>11}"
        f"{'TODAY ACC':>12}"
    )

    for result in results:
        print(
            f"{result['model']:<24}"
            f"{result['accuracy'] * 100:>11.2f}%"
            f"{result['macro_f1'] * 100:>11.2f}%"
            f"{result['session_accuracy'] * 100:>13.2f}%"
            f"{result['old_accuracy'] * 100:>10.2f}%"
            f"{result['today_accuracy'] * 100:>11.2f}%"
        )

    best = max(
        results,
        key=lambda r: (
            r["macro_f1"],
            r["accuracy"],
        )
    )

    print()
    print(
        f"Best by Macro F1 : "
        f"{best['model']}"
    )


if __name__ == "__main__":
    main()
