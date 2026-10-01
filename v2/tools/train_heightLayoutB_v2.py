import json
import hashlib
from pathlib import Path

import joblib
import numpy as np

from train_activity_baseline import build_model
from validate_heightLayoutA_cv import read_frames, build_windows


SESSIONS = [
    # OLD block
    ("SITTING", "sitting_heightLayoutB_stable_001"),
    ("SITTING", "sitting_heightLayoutB_stable_002"),
    ("SITTING", "sitting_heightLayoutB_stable_003"),

    ("MOVING", "moving_heightLayoutB_stable_001"),
    ("MOVING", "moving_heightLayoutB_stable_002"),
    ("MOVING", "moving_heightLayoutB_stable_003"),

    ("LYING", "lying_heightLayoutB_stable_001"),
    ("LYING", "lying_heightLayoutB_stable_002"),
    ("LYING", "lying_heightLayoutB_stable_003"),

    # 2026-09-08 block
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


MODEL_DIR = Path("data/models/activity")
MODEL_PATH = MODEL_DIR / "activity_heightLayoutB_v2.joblib"
META_PATH = MODEL_DIR / "activity_heightLayoutB_v2.json"


def main():
    forbidden = "lying_heightLayoutB_stable_repeat_20260908_02"

    session_ids = [sid for _, sid in SESSIONS]

    if len(session_ids) != 18:
        raise RuntimeError(
            f"Expected 18 sessions, got {len(session_ids)}"
        )

    if len(set(session_ids)) != 18:
        raise RuntimeError("Duplicate session detected")

    if forbidden in session_ids:
        raise RuntimeError(
            "Mislabeled 3.84 sec LYING file must not be used"
        )

    x_train = []
    y_train = []

    class_windows = {
        "SITTING": 0,
        "MOVING": 0,
        "LYING": 0,
    }

    print("=" * 80)
    print("TRAIN activity_heightLayoutB_v2")
    print("18 sessions / 2 time blocks / RANDOM_FOREST")
    print("=" * 80)

    for label, sid in SESSIONS:
        frames = read_frames(label, sid)
        windows = build_windows(frames)

        print(
            f"{sid:<55} "
            f"{label:<7} "
            f"frames={len(frames):>4} "
            f"windows={len(windows):>2}"
        )

        for feature in windows:
            x_train.append(feature)
            y_train.append(label)

        class_windows[label] += len(windows)

    x_train = np.asarray(
        x_train,
        dtype=np.float64,
    )

    y_train = np.asarray(y_train)

    print()
    print("Training summary")
    print(f"Sessions      : {len(SESSIONS)}")
    print(f"Total windows : {len(x_train)}")
    print(f"Feature dim   : {x_train.shape[1]}")

    for label in ["SITTING", "MOVING", "LYING"]:
        print(
            f"{label:<7} windows : "
            f"{class_windows[label]}"
        )

    if x_train.shape[1] != 918:
        raise RuntimeError(
            f"Expected 918 features, got {x_train.shape[1]}"
        )

    print()
    print("Training Random Forest...")

    model = build_model("RANDOM_FOREST")
    model.fit(x_train, y_train)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    # 저장한 파일을 다시 읽을 수 있는지 확인
    loaded_model = joblib.load(MODEL_PATH)

    test_pred = loaded_model.predict(
        x_train[:3]
    )

    if len(test_pred) != 3:
        raise RuntimeError(
            "Saved model reload check failed"
        )

    sha256 = hashlib.sha256(
        MODEL_PATH.read_bytes()
    ).hexdigest()

    metadata = {
        "model_id": "activity_heightLayoutB_v2",
        "model_type": "RANDOM_FOREST",
        "classes": [
            "SITTING",
            "MOVING",
            "LYING",
        ],
        "training_sessions": 18,
        "training_windows": int(len(x_train)),
        "class_windows": class_windows,
        "feature_dimension": int(x_train.shape[1]),
        "window_seconds": 3.0,
        "hop_seconds": 1.5,
        "feature_definition": (
            "306 temporal mean + "
            "306 population std + "
            "306 mean absolute temporal diff"
        ),
        "time_blocks": [
            "heightLayoutB_stable original",
            "2026-09-08",
        ],
        "model_selection_cv": {
            "window_accuracy": 0.9238,
            "macro_f1": 0.9214,
            "session_accuracy": 0.9444,
        },
        "sha256": sha256,
        "note": (
            "2026-09-08 sessions are now training data. "
            "Future unseen date data must remain test-only "
            "until its evaluation is recorded."
        ),
    }

    META_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("#" * 80)
    print("MODEL SAVED")
    print("#" * 80)
    print(f"Model : {MODEL_PATH}")
    print(f"Meta  : {META_PATH}")
    print(f"SHA256: {sha256}")
    print()
    print("Reload check : PASS")


if __name__ == "__main__":
    main()
