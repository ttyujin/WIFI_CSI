# Wi-Fi CSI 생활 상태 모니터링 연구 서비스

이 문서는 현재 checkout의 코드를 기준으로 연구 서비스와 원본 RuView의
역할을 구분한다. 원본 기능의 삭제나 별도 아키텍처로의 재작성은 수행하지 않는다.
원본 저작권·라이선스·하위 모듈 안내는 저장소에 그대로 유지한다.

## 현재 실행 경로

| 단계 | 실제 파일/경로 | 역할 |
|---|---|---|
| ESP32 | `firmware/esp32-csi-node/main/csi_collector.c`, `stream_sender.c` | CSI callback, sequence, ADR-018 데이터 직렬화와 UDP 전송 |
| Rust 수신 | `v2/crates/wifi-densepose-sensing-server/src/main.rs` | ESP32 UDP 파싱·처리, HTTP/WS 서버와 라우터 구성 |
| CSI 전달 계약 | `v2/crates/wifi-densepose-sensing-server/src/activity_recording.rs` | `ActivityCsiSample`, `is_recordable_csi`, 실제 306-subcarrier 프레임 검증 |
| 연구 WebSocket | `main.rs`의 `handle_ws_activity_csi_client` | `activity_csi_tx`의 유효 CSI를 `/ws/activity/csi`로 전달 |
| Python 추론/API | `v2/tools/activity_app_server.py` | WS 연결, timestamp window, 모델 추론, 상태·history, HTTP API |
| 실제 특징 추출 의존성 | `validate_heightLayoutA_cv.py` → `train_activity_baseline.py` → `prepare_activity_dataset.py` (`v2/tools/`) | 기존 추론에서 import하는 함수·상수·모듈. 파일 이름이 예전 실험이어도 삭제 금지 |
| 현재 모델 | `v2/data/models/activity/activity_heightLayoutB_v2.joblib` | 3-class Random Forest; `predict_proba`의 MOVING 확률 사용 |
| 이메일/프로필 | `v2/tools/senior_monitor_alerts.py` | 프로필 저장, duration/stage 중복 방지, 전용 SMTP worker |
| React 진입점 | `ui/senior-monitor/index.html`, `src/main.tsx`, `src/App.tsx` | Vite + React + TypeScript 보호자 웹 |
| 웹 API/연결 상태 | `src/api/activityApi.ts`, `profileApi.ts`, `src/hooks/useActivity.ts` | 현재 hostname의 8010 API 조회, 프로필, 연결 상태 |
| 표시/합계/캐릭터 | `src/components/`, `src/utils/`, `src/styles/app.css`, `images/` | 현재 상태·시간·history·프로필·반응형 UI 및 기존 GIF |

이메일 발송은 Python의 정상 prediction에서 시작한다. React polling이나
브라우저 타이머가 메일을 발송하지 않는다.

### 모델 및 시간 처리 계약

- 입력은 실제 유효 CSI의 306개 amplitude다. status/vitals와 불완전한 프레임은
  연구 채널과 Python의 유효 프레임 처리에서 제외된다.
- window = 3.0초, hop = 1.5초. Python은 timestamp의 반열린 구간 `[start, end)`을 사용한다.
- 각 frame을 그 frame의 amplitude 평균으로 나눈 뒤 서브캐리어별 시간 평균,
  표준편차, 연속 frame 간 평균 절대 변화량을 그 순서로 연결한다: 306 × 3 = 918.
- `abs(frame_mean) <= 1e-12`인 frame은 기존 함수에서 0 벡터로 처리한다.
- 모델 클래스는 LYING/MOVING/SITTING이고, `P(MOVING) >= 0.70`이면 MOVING,
  그 미만이면 STAYING이다. 별도의 2-class 재학습을 의미하지 않는다.
- Python의 초기 `websocket.create_connection(..., timeout=5)`와 연결 후
  `ws.settimeout(1)`은 서로 다른 목적이다. 기존 연결 장애 해결값을 유지한다.
- CSI 또는 prediction freshness가 5초를 넘으면 연결 상태를 전환한다.
  단절 구간과 복구 후 window 준비시간은 활동 시간에 포함하지 않는다.
- history는 메모리에 유지한다. Python을 재시작하면 이전 활동 기록은 복구되지 않는다.
  개인 프로필은 `v2/data/senior-monitor/profile.json`에 별도로 저장한다.

## Clone 후 준비할 것

1. 현재 연구 코드 변경분이 포함된 저장소를 사용한다. 로컬의 untracked 파일은
   commit되기 전까지 GitHub clone에 포함되지 않는다.
2. Rust 소스를 빌드할 때는 저장소의 submodule을 준비한다.
   `v2/Cargo.toml`의 path dependency와 `.gitmodules`를 유지해야 한다.

   ```powershell
   git submodule update --init --recursive
   ```

3. `v2/rust-toolchain.toml`의 Rust toolchain과 Windows 빌드 도구를 준비한다.
4. 기존 모델과 호환되는 Python 환경에서 다음 의존성을 준비한다.

   ```powershell
   # RuView/v2
   python -m pip install -r tools/activity_baseline_requirements.txt joblib websocket-client
   ```

   requirements의 numpy/scikit-learn 버전을 임의로 바꾸지 않는다. 운영 중인
   Python 환경은 보존하고, 새 개발 환경을 준비하는 경우에만 설치한다.
5. Node.js는 `ui/senior-monitor/package.json`의 engine 조건(24 이상)을 따른다.
   `ui/senior-monitor`에서 `npm ci`로 lockfile 기반 의존성을 준비한다.
6. 현재 모델을 별도로 받아 정확히
   `v2/data/models/activity/activity_heightLayoutB_v2.joblib`에 준비한다.
   `.gitignore`의 `models/` 규칙 때문에 이 모델은 자동으로 clone되지 않는다.
   이 위치의 파일을 임의 모델로 대체하거나 이전 `models/activity_v1` artifact와 혼동하지 않는다.
7. 실측 recording, processed 연구 데이터, 개인 프로필, credential, `target/`의
   실행 파일은 소스와 구분한다. 기존 로컬 파일을 삭제하지 않는다.

## 실행 순서와 포트

현재 연구 PC의 LAN IP는 `192.168.0.60`이다. 다른 PC에서는 실제 IP에 맞는
ESP32 전송 대상과 UDP bind/allow 설정을 준비해야 한다.
아래는 현재 PC의 기존 실행 설정이며 자동으로 IP/포트를 변경하지 않는다.

| 용도 | 현재 주소/포트 |
|---|---|
| ESP32 → Rust | UDP `192.168.0.60:5005` |
| RuView HTTP | `3000` |
| 연구 CSI WebSocket | `ws://localhost:3001/ws/activity/csi` |
| Python API | bind `0.0.0.0:8010` |
| Vite 보호자 웹 | bind `0.0.0.0:8090`, strict port |
| 휴대폰 브라우저 | `http://192.168.0.60:8090` |
| SMTP | `smtp.gmail.com:465` (SSL) |

### 터미널 1: 기존 Rust 바이너리

```powershell
cd C:\Users\user\Desktop\wifi\RuView\v2

.\target\activity-live-v2-build\release\sensing-server.exe `
  --source esp32 `
  --udp-port 5005 `
  --udp-bind 192.168.0.60 `
  --udp-allow 192.168.0.0/24 `
  --http-port 3000 `
  --ws-port 3001 `
  --ui-path ..\ui
```

`--ui-path ..\ui`는 기존 RuView 정적 UI 경로다. 보호자 React 웹은 별도의
Vite 프로세스에서 8090으로 제공된다. 따라서 `ui/senior-monitor`만 남기고
원본 `ui`를 삭제해서는 안 된다.

clone한 환경에 실행 파일이 없다면 별도 target dir로 소스 빌드한다.
현재 실행 중인 파일을 교체하는 작업과는 구분한다.

```powershell
# RuView/v2
cargo build --locked -p wifi-densepose-sensing-server --bin sensing-server `
  --release --target-dir target/activity-live-v2-build
```

### 터미널 2: Python 및 선택적 이메일 설정

```powershell
cd C:\Users\user\Desktop\wifi\RuView\v2
python tools/activity_app_server.py
```

메일 credential은 이 Python 프로세스가 시작되기 전에 같은 터미널의
환경변수에 설정한다. credential이 없으면 이메일만 비활성화되고 CSI/API는 동작한다.
실제 비밀번호를 파일·README·명령문에 직접 넣지 않는다.
안전한 입력 절차는 [EMAIL_ALERTS.md](../ui/senior-monitor/EMAIL_ALERTS.md)를 따른다.

| 환경변수 | 소스 기본값/역할 |
|---|---|
| `WIFI_ELDER_SENDER_EMAIL` | 발신 주소; 코드에 있는 기존 기본 주소 또는 운영자가 설정한 주소 |
| `WIFI_ELDER_SENDER_APP_PASSWORD` | Gmail App Password; 기본 비밀번호 없음 |
| `WIFI_ELDER_STAYING_NOTICE_SECONDS` | 10800초 (3시간) |
| `WIFI_ELDER_MOVING_NOTICE_SECONDS` | 3600초 (1시간) |
| `WIFI_ELDER_CAUTION_SECONDS` | 14400초 (4시간) |
| `WIFI_ELDER_DANGER_SECONDS` | 21600초 (6시간) |

이 값은 현재 소스의 변경 가능한 서비스 기본값이며 의료적 판단 기준이 아니다.
운영 환경변수로 덮어쓸 수 있으므로 실제 실행 설정과 구분해야 한다.
기존 `WIFI_ELDER_STAYING_ALERT_SECONDS` 및 `WIFI_ELDER_MOVING_ALERT_SECONDS`는
해당 NOTICE 변수가 없을 때 읽는 호환용 이름이다. 각 상태의 NOTICE < CAUTION < DANGER여야 한다.
유효 프로필과 설정이 필요하며, SMTP 실패 시 같은 interval/stage를 자동 재전송하지 않는다.

### 터미널 3: 보호자 웹

```powershell
cd C:\Users\user\Desktop\wifi\RuView\ui\senior-monitor
npm ci
npm start
```

PC: `http://localhost:8090`. 같은 LAN의 휴대폰: `http://192.168.0.60:8090`.
웹은 `http://${window.location.hostname}:8010`에 접근한다. 모바일용 API를
`localhost:8010`으로 고정하지 않는다. 방화벽/LAN 연결은 기존 설정을 유지한다.

## API와 연구 도구

- Python: `GET /health`, `GET /api/activity/current`,
  `GET /api/activity/current?include_history=1`, `GET /api/activity/history`,
  `GET /api/profile`, `POST /api/profile`.
- Rust: 기존 sensing/nodes/recording API와 `/ws/activity/csi`를 유지한다.
- 연구용 수집: `v2/tools/collect_activity.ps1`, `activity_qc.py`,
  Rust의 `/api/v1/activity/recording/start` 및 `/stop`.
- 연구 데이터 audit/학습/평가: `prepare_activity_dataset.py`,
  `analyze_activity_bimodality.py`, `train_activity_baseline.py`,
  `train_heightLayoutB_v2.py`, `validate_*`, `evaluate_*`, `compare_*`, `inspect_*`.
- 이전 artifact와 CLI 추론: `build_activity_v1_model.py`, `activity_inference.py`,
  `live_activity_inference.py`. 현재 백엔드와 동일한 진입점이라는 뜻은 아니지만,
  연구 재현·데이터 파이프라인과 관련되어 보존한다.

## 왜 원본 RuView 코드가 남아 있는가

`main.rs`는 연구 WS 외에도 signal, pose tracker, vitals, engine/field bridge,
모델 API 등을 같은 바이너리에 연결한다. `lib.rs` 역시 여러 모듈을 공개하며,
Cargo의 path dependency에는 hardware/signal/core/engine/physics/worldgraph 등과
`vendor/rufield`가 연결된다. runtime에서 특정 서비스를 사용하지 않는다는
이유만으로 관련 `.rs`나 crate를 삭제할 수 없다.

`archive/v1`은 원본 Python CI·Docker 참조가 있고, 원본 `ui`는 기존 서버의
정적 파일 경로와 JavaScript 회귀 테스트에 사용된다. examples, scripts,
assets, vendor, harness 및 원본 문서는 빌드·배포·테스트·설명 참조가 남아 있다.
이들은 보호자 서비스의 직접 실행 경로와 구분해서 읽되 삭제 가능하다고 추정하지 않는다.

## 검증 명령

```powershell
# RuView repository root: backend/profile/email (SMTP mock, 실메일 발송 없음)
python -m unittest discover -s v2/tools -p "test_*.py" -v
python -m py_compile v2/tools/activity_app_server.py v2/tools/senior_monitor_alerts.py

# RuView/v2: dataset, QC, training/artifact tests
python -m unittest discover -s tools/tests -p "test_*.py" -v
.\tools\tests\test_collect_activity.ps1

# RuView/ui/senior-monitor
npm test
npm run build

# RuView/v2: 기존 실행 바이너리를 덮어쓰지 않는 check/test
cargo check --locked -p wifi-densepose-sensing-server --bin sensing-server `
  --target-dir target/activity-cleanup-check
cargo test --locked -p wifi-densepose-sensing-server `
  --target-dir target/activity-cleanup-check
```

단위 테스트의 CSI와 SMTP는 fixture/mock이다. 실제 ESP32 연결·모델 정확도·
Gmail 수신을 입증하는 결과로 표현하지 않는다. 상세 수동 확인 절차는
[보호자 웹 README](../ui/senior-monitor/README.md)에 있다.
