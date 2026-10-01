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
    # Fold 1: old 001 + today original holdout
    [
        ("OLD", "SITTING", "sitting_heightLayoutB_stable_001"),
        ("OLD", "MOVING",  "moving_heightLayoutB_stable_001"),
        ("OLD", "LYING",   "lying_heightLayoutB_stable_001"),

        ("TODAY", "SITTING", "sitting_heightLayoutB_stable_holdout_20260908"),
        ("TODAY", "MOVING",  "moving_heightLayoutB_stable_holdout_20260908"),
        ("TODAY", "LYING",   "lying_heightLayoutB_stable_holdout_20260908"),
    ],

    # Fold 2: old 002 + today repeat 01
    [
        ("OLD", "SITTING", "sitting_heightLayoutB_stable_002"),
        ("OLD", "MOVING",  "moving_heightLayoutB_stable_002"),
        ("OLD", "LYING",   "lying_heightLayoutB_stable_002"),

        ("TODAY", "SITTING", "sitting_heightLayoutB_stable_repeat_20260908_01"),
        ("TODAY", "MOVING",  "moving_heightLayoutB_stable_repeat_20260908_01"),
        ("TODAY", "LYING",   "lying_heightLayoutB_stable_repeat_20260908_01"),
    ],

    # Fold 3: old 003 + today repeat 02
    [
        ("OLD", "SITTING", "sitting_heightLayoutB_stable_003"),
        ("OLD", "MOVING",  "moving_heightLayoutB_stable_003"),
        ("OLD", "LYING",   "lying_heightLayoutB_stable_003"),

        ("TODAY", "SITTING", "sitting_heightLayoutB_stable_repeat_20260908_02"),
        ("TODAY", "MOVING",  "moving_heightLayoutB_stable_repeat_20260908_02"),
        ("TODAY", "LYING",   "lying_heightLayoutB_stable_repeat_20260908_02_retry"),
    ],
]


def main():

    # ------------------------------------------------------------
    # 정확히 18개 세션 구성
    # ------------------------------------------------------------
    all_specs = []

    for fold in FOLDS:
        all_specs.extend(fold)

    session_ids = [sid for _, _, sid in all_specs]

    if len(session_ids) != 18:
        raise RuntimeError(
            f"Expected 18 sessions, got {len(session_ids)}"
        )

    if len(set(session_ids)) != 18:
        raise RuntimeError("Duplicate session detected")

    # 잘못 찍힌 3.84초 LYING 파일 방지
    forbidden = "lying_heightLayoutB_stable_repeat_20260908_02"

    if forbidden in session_ids:
        raise RuntimeError(
            "Mislabeled 3.84 sec LYING file must not be used"
        )

    # ------------------------------------------------------------
    # 전체 세션 미리 로딩
    # ------------------------------------------------------------
    sessions = {}

    print("=" * 82)
    print("LOAD COMBINED TWO-TIME-BLOCK DATASET")
    print("OLD 9 sessions + 2026-09-08 9 sessions = 18 sessions")
    print("=" * 82)

    for block, label, sid in all_specs:

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
            f"{block:<6} "
            f"{sid:<52} "
            f"label={label:<7} "
            f"frames={len(frames):>4} "
            f"windows={len(windows):>2}"
        )

    # ------------------------------------------------------------
    # OOF 결과
    # ------------------------------------------------------------
    all_true = []
    all_pred = []
    all_block = []

    session_correct = 0
    session_total = 0

    fold_accs = []
    fold_f1s = []

    print()
    print("=" * 82)
    print("COMBINED-BLOCK 3-FOLD SESSION-LEVEL CV")
    print("918 normalized features")
    print("StandardScaler + LogisticRegression")
    print("NO SESSION LEAKAGE")
    print("=" * 82)

    for fold_idx, test_specs in enumerate(FOLDS, start=1):

        test_ids = {
            sid
            for _, _, sid in test_specs
        }

        # --------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------
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

        model = build_model(
            "LOGISTIC_REGRESSION"
        )

        model.fit(
            x_train,
            y_train,
        )

        # --------------------------------------------------------
        # TEST
        # --------------------------------------------------------
        fold_true = []
        fold_pred = []

        print()
        print("-" * 82)
        print(f"FOLD {fold_idx}")
        print("-" * 82)

        for block, actual_label, sid in test_specs:

            x_test = sessions[sid]["windows"]

            pred = model.predict(x_test)
            proba = model.predict_proba(x_test)

            model_classes = [
                str(c)
                for c in model.classes_
            ]

            mean_probs = {
                label: float(
                    proba[
                        :,
                        model_classes.index(label)
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
            print(
                f"[{block}] {sid}"
            )
            print(
                f"Actual  : {actual_label}"
            )
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

            all_block.extend(
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

        fold_accs.append(fold_acc)
        fold_f1s.append(fold_f1)

        print()
        print(
            f"Fold window accuracy : "
            f"{fold_acc * 100:.2f}%"
        )
        print(
            f"Fold macro F1        : "
            f"{fold_f1 * 100:.2f}%"
        )

    # ------------------------------------------------------------
    # 전체 OOF
    # ------------------------------------------------------------
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
    print("#" * 82)
    print("FINAL COMBINED TWO-BLOCK CV RESULT")
    print("#" * 82)

    print(
        f"Fold accuracy mean : "
        f"{np.mean(fold_accs) * 100:.2f} "
        f"± {np.std(fold_accs) * 100:.2f}%"
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
    print("Combined OOF confusion matrix")
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

    # ------------------------------------------------------------
    # OLD / TODAY 별도 성능도 확인
    # ------------------------------------------------------------
    all_true_np = np.asarray(all_true)
    all_pred_np = np.asarray(all_pred)
    all_block_np = np.asarray(all_block)

    print()
    print("#" * 82)
    print("PER-TIME-BLOCK OOF RESULT")
    print("#" * 82)

    for block in ["OLD", "TODAY"]:

        mask = (
            all_block_np == block
        )

        block_true = all_true_np[mask]
        block_pred = all_pred_np[mask]

        block_acc = accuracy_score(
            block_true,
            block_pred,
        )

        _, _, block_f1, _ = (
            precision_recall_fscore_support(
                block_true,
                block_pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        )

        block_matrix = confusion_matrix(
            block_true,
            block_pred,
            labels=LABELS,
        )

        print()
        print(
            f"{block} window accuracy : "
            f"{block_acc * 100:.2f}%"
        )

        print(
            f"{block} macro F1        : "
            f"{block_f1 * 100:.2f}%"
        )

        print(
            f"{block} confusion matrix"
        )

        print(
            "Actual \\ Pred     "
            "SITTING   MOVING   LYING"
        )

        for label, row in zip(
            LABELS,
            block_matrix,
        ):
            print(
                f"{label:<17}"
                f"{row[0]:>7}"
                f"{row[1]:>9}"
                f"{row[2]:>8}"
            )


if __name__ == "__main__":
    main()
