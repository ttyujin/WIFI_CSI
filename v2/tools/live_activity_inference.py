import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import websocket


WS_URL = "ws://localhost:3001/ws/activity/csi"

WINDOW_SEC = 3.0
HOP_SEC = 1.5
EXPECTED_SUBCARRIERS = 306

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "models"
    / "activity"
    / "activity_heightLayoutB_v2.joblib"
)


def make_feature(amplitudes):
    x = np.asarray(amplitudes, dtype=np.float64)

    frame_mean = x.mean(axis=1, keepdims=True)

    if np.any(np.abs(frame_mean) < 1e-12):
        raise ValueError("frame mean is zero")

    # 학습 때와 동일한 frame-wise mean normalization
    x = x / frame_mean

    temporal_mean = x.mean(axis=0)
    temporal_std = x.std(axis=0, ddof=0)

    if len(x) < 2:
        raise ValueError("not enough frames")

    temporal_diff = np.abs(
        np.diff(x, axis=0)
    ).mean(axis=0)

    feature = np.concatenate([
        temporal_mean,
        temporal_std,
        temporal_diff,
    ])

    if feature.shape[0] != 918:
        raise ValueError(
            f"feature length must be 918, got {feature.shape[0]}"
        )

    return feature


def connect_websocket():
    while True:
        try:
            print("Connecting to CSI WebSocket...")

            ws = websocket.create_connection(
                WS_URL,
                timeout=10,
            )

            print("CONNECTED")
            print("최근 3초 CSI를 모으는 중...")
            print()

            return ws

        except KeyboardInterrupt:
            raise

        except Exception as e:
            print(
                "WebSocket connect failed:",
                type(e).__name__,
                str(e),
            )
            print("2초 후 다시 연결합니다...")
            time.sleep(2)


print("=" * 60)
print("RuView Live Activity Inference")
print("=" * 60)
print("Model :", MODEL_PATH)
print("WS    :", WS_URL)

model = joblib.load(MODEL_PATH)

print("Classes :", model.classes_)
print("Features:", model.n_features_in_)
print()


frames = deque()

first_timestamp = None
next_end = None
last_timestamp = None

ws = None


def reset_buffer():
    global first_timestamp, next_end, last_timestamp

    frames.clear()
    first_timestamp = None
    next_end = None
    last_timestamp = None


try:

    while True:

        # 연결이 없으면 새 WebSocket 생성
        if ws is None:
            ws = connect_websocket()
            reset_buffer()

        try:
            message = ws.recv()

        except websocket.WebSocketTimeoutException:

            print()
            print(
                "CSI WebSocket timeout -> reconnecting..."
            )

            try:
                ws.close()
            except Exception:
                pass

            ws = None
            reset_buffer()

            time.sleep(1)
            continue

        except websocket.WebSocketConnectionClosedException:

            print()
            print(
                "WebSocket closed -> reconnecting..."
            )

            try:
                ws.close()
            except Exception:
                pass

            ws = None
            reset_buffer()

            time.sleep(1)
            continue

        except OSError as e:

            print()
            print(
                "WebSocket network error -> reconnecting:",
                e,
            )

            try:
                ws.close()
            except Exception:
                pass

            ws = None
            reset_buffer()

            time.sleep(1)
            continue


        try:
            data = json.loads(message)

        except json.JSONDecodeError:
            continue


        if data.get("type") != "activity_csi":
            continue


        timestamp = float(data["timestamp"])
        sequence = int(data["sequence"])
        amplitude = data["amplitude"]


        if len(amplitude) != EXPECTED_SUBCARRIERS:
            continue


        arr = np.asarray(
            amplitude,
            dtype=np.float64,
        )


        if not np.all(np.isfinite(arr)):
            continue


        # ESP32 timestamp가 뒤로 가면
        # 새 stream으로 판단하고 버퍼 초기화
        if (
            last_timestamp is not None
            and timestamp < last_timestamp
        ):
            print()
            print(
                "CSI timestamp reset detected"
                " -> buffer reset"
            )

            reset_buffer()


        last_timestamp = timestamp


        frames.append(
            (
                timestamp,
                sequence,
                arr,
            )
        )


        if first_timestamp is None:
            first_timestamp = timestamp
            next_end = (
                first_timestamp
                + WINDOW_SEC
            )


        # 최근 데이터만 유지
        while (
            frames
            and frames[0][0]
            < timestamp - 6.0
        ):
            frames.popleft()


        # 최초 3초가 아직 안 찼음
        if timestamp < next_end:
            continue


        # 1.5초마다 예측
        while timestamp >= next_end:

            window_start = (
                next_end - WINDOW_SEC
            )

            window = [
                frame
                for frame in frames
                if (
                    window_start
                    <= frame[0]
                    < next_end
                )
            ]


            if len(window) >= 2:

                amplitudes = [
                    frame[2]
                    for frame in window
                ]

                try:

                    feature = make_feature(
                        amplitudes
                    )

                    probabilities = (
                        model.predict_proba(
                            feature.reshape(1, -1)
                        )[0]
                    )

                    best_index = int(
                        np.argmax(probabilities)
                    )

                    predicted = (
                        model.classes_[
                            best_index
                        ]
                    )

                    confidence = float(
                        probabilities[
                            best_index
                        ]
                    )

                    prob_map = {
                        label: float(prob)
                        for label, prob
                        in zip(
                            model.classes_,
                            probabilities,
                        )
                    }

                    moving_prob = prob_map.get("MOVING", 0.0)

                    if moving_prob >= 0.90:
                        final_state = "MOVING"
                    else:
                        final_state = "STAYING"

                    now = (
                        datetime.now()
                        .strftime("%H:%M:%S")
                    )

                    print(
                        f"{now}  "
                        f"{final_state:<8} "
                        f"M={moving_prob:.3f}  "
                        f"frames={len(window):3d}  "
                        f"seq={sequence}"
                    )

                    print(
                        "          "
                        f"L={prob_map.get('LYING', 0):.3f}  "
                        f"M={prob_map.get('MOVING', 0):.3f}  "
                        f"S={prob_map.get('SITTING', 0):.3f}"
                    )

                except Exception as e:
                    print(
                        "Prediction error:",
                        e,
                    )


            next_end += HOP_SEC


except KeyboardInterrupt:

    print()
    print("Stopped by user.")


finally:

    if ws is not None:
        try:
            ws.close()
        except Exception:
            pass
