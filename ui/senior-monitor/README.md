# Senior Monitor

React + TypeScript + Vite 기반 보호자용 화면입니다. 기존 RuView UI/Rust,
306 CSI parsing/recording, 연구 데이터와 모델은 변경하지 않습니다.

## 실행

이메일 설정/API/수동 Gmail 1회 검증은 [EMAIL_ALERTS.md](EMAIL_ALERTS.md)를 참고하세요.

기존 모델 실행용 Python 환경에서 backend를 재시작하고 frontend도 함께 갱신하세요.
구형 backend는 명시적 연결 상태 필드가 없어 새 UI가 안전하게 거부합니다.

```powershell
# 터미널 1: 기존에 정상 수신하던 RuView 실행 명령/UDP 허용 IP를 그대로 사용.
# HTTP 3000, WS 3001, UDP 5005 설정을 유지합니다.
# 현재 존재하는 최신 별도 빌드:
# C:\Users\user\Desktop\wifi\RuView\v2\target\activity-live-v2-build\release\sensing-server.exe

# 터미널 2: 기존 joblib/numpy/scikit-learn/websocket-client 환경
cd C:\Users\user\Desktop\wifi\RuView\v2
python tools/activity_app_server.py

# 터미널 3: Node.js 24 이상
cd C:\Users\user\Desktop\wifi\RuView\ui\senior-monitor
npm install
npm start
```

브라우저: http://localhost:8090
API: 현재 페이지 hostname의 HTTP 8010 (기존 LAN 접근 방식 유지).
사용 중인 8090 포트를 자동 변경하지 않습니다.

## 상태와 시간의 단일 기준

- 모델/feature extraction은 기존 그대로. 최종 서비스 상태는 MOVING/STAYING만 사용.
- backend의 확정 기준: P(MOVING) >= 0.70이면 MOVING, 미만이면 STAYING.
- CONNECTING: 최초 데이터 대기. socket 연결 성공만으로 LIVE가 되지 않음.
- LIVE: 실제 valid306 CSI로 새 3초 timestamp window를 만든 뒤 정상 prediction 성공.
- RECONNECTING: socket 오류, 유효 CSI 또는 LIVE prediction이 5초간 없음.
- 유효 CSI는 Rust의 activity_csi 계약, valid node/sequence/timestamp,
  subcarrier_count/amplitude_count/실제 배열 길이 모두306, 숫자·유한 amplitude.
  status/vitals, malformed frame, 중복 timestamp는 freshness를 유지하지 않음.
- 중앙 StreamState가 monotonic clock으로 freshness를 판단함. HTTP 읽기와
  receive-loop의 1초 wakeup은 같은 정책을 호출함(별도 timeout 판정 아님).
- 단절 때 마지막 성공 prediction timestamp에서 현재 구간을 한 번만 종료.
  UNKNOWN과 단절 구간은 history에 저장하지 않음.
- 복구 시 이전 window 폐기. 첫 새 prediction부터 새 interval을 시작.
  이전 generation의 늦은 prediction은 LIVE를 복원할 수 없음.
- current?include_history=1은 같은 lock 안에서 current/history를 함께 복사.
  기존 current 기본 응답 및 history 배열 endpoint도 유지.
- frontend는 이 atomic snapshot을 응답 완료 후 1초마다 조회.
  겹치는 poll 없음, HTTP timeout4초, cleanup 시 abort/모든 timer 정리.
  구 effect/timeout 이후 늦은 응답은 무시.
- HTTP 접근 가능 여부와 backend CSI 상태를 구분. HTTP 성공 자체는 LIVE 아님.
  API 실패/RECONNECTING이면 정확히 “다시 연결 중”, GIF/현재 구간/타이머 없음.
- frontend의 과거10초 stale 추정은 제거. CSI freshness의 기준은 backend뿐.
- duration_seconds는 부드러운 표시를 위해 최대1 hop(1.5초) 지연.
  confirmed_duration_seconds(마지막 성공 prediction까지) 이상으로 보간하지 않음.
  따라서 단절 감지까지 기다리는5초도 가짜 시간으로 추가되지 않음.
- 합계는 완료 history + 현재 LIVE 구간만 사용. UNKNOWN/HTTP 실패 시 열린
  구간을 제외(backend가 종료를 확정해 보내면 완료 기록으로 다시 반영).
  API 장애 때 화면이 임의로 완료 기록을 만들어내지 않음.

## 프로필 / 오늘 합계

- backend의 v2/data/senior-monitor/profile.json이 authoritative profile입니다.
  name/gender/guardian_email만 저장하며 localStorage는 UI 캐시입니다.
- 최초 설정은 이름 → 성별 → 보호자 이메일의3단계 wizard. 이메일이 없는 기존
  name/characterType 캐시는 이름·성별을 보존하고 이메일 단계부터 이어집니다.
- backend 저장 성공 후에만 완료합니다. 캐시 저장 실패는 서버 저장을 취소하지 않고
  경고를 표시하며 다음 접속에서 backend 값으로 복구합니다.
- 최초 설정/재진입/프로필 수정/할머니·할아버지와 네 GIF 유지.
- localStorage는 origin별. 프로필 수정은 polling을 재시작하지 않음.
- GIF 원본과 이름(예: grandma-sitaying.gif) 변경 없음.
- 완료 구간의 ISO timestamp를 브라우저 로컬의 오늘 범위와 교차해 합산.
  UTC 날짜/고정24시간 계산을 사용하지 않아 자정/DST에도 구간 보존.
- 완료 목록은 최근15개지만 합계는 전체 history.
- 중복/겹친 동종 구간은 중복 합산하지 않음.

## 자동 검증

```powershell
# ui/senior-monitor
npm test
npm run typecheck -- --noUnusedLocals --noUnusedParameters
npm run build

# RuView repository root (모델 의존성 없이 표준 Python으로 실행 가능)
python -m py_compile v2/tools/activity_app_server.py v2/tools/test_activity_app_server.py
python -m unittest discover -s v2/tools -p test_activity_app_server.py -v
python -m unittest discover -s v2/tools -p test_senior_monitor_alerts.py -v
```

테스트의 CSI/시계/확률은 SYNTHETIC fixture이며 연구 recording에 쓰지 않습니다.
빌드는 dist/에 생성하며 원본 GIF 네 개를 포함합니다.
Windows sandbox worker 시작 timeout 발생 시 코드 assertion 실패와 구분해야 합니다.

## 실제 ESP32 수동 확인 (자동/하드웨어 실험 결과가 아닌 사용자용 절차)

1. 정상 동작하던 ESP32 연결과 RuView 실행 옵션을 유지해 서버 실행.
   HTTP3000, WS3001/ws/activity/csi, UDP5005 확인. UDP 허용 IP를 임의 변경하지 않음.
2. 기존 모델 의존성이 설치된 Python 환경에서 위 backend 실행.
3. npm start 후 localhost:8090 접속. 기존 프로필 또는 최초 설정 확인.
4. valid306 CSI의 3초 window 후 “실시간 연결됨”, 실제 활동 상태 확인.
5. STAYING/MOVING 각각 timer와 해당 합계가 증가하는지 확인.
6. ESP32 USB 제거. 최대 약5초 + HTTP poll/응답 지연 후 “다시 연결 중” 확인.
7. GIF/진행 중 기록이 사라지고, 완료 기록 끝이 last_prediction_at인지 확인.
   timer/오늘 합계가 늘지 않는지20초 대기하며 확인.
8. USB 재연결. socket 연결만으로 LIVE가 되지 않고 새3초 window 이후 복귀하는지 확인.
9. 이전 상태와 같더라도 새 started_at/0초 interval로 시작하는지 확인.
10. 단절20초 및 복구 window 준비시간이 두 활동 합계/history에 없는지 확인.
11. Python API를 Ctrl+C로 종료해도 같은 “다시 연결 중”과 시간 증가 중단 확인.
12. Python 재시작 후 HTTP 응답만으로 LIVE 표시하지 않는지 확인.

## 제한사항

- 메모리 history: Python 재시작 전 기록은 복구하지 않음. UI에도 범위를 명시.
- backend와 frontend를 함께 갱신해야 함. 기존 old current payload를 LIVE로 추정하지 않음.
- 표시 시간은 최대1.5초 늦고, 단절 감지/화면 갱신에는 freshness5초와 poll 지연이 있음.
- 실제 ESP32 재연결/모델 추론 성능은 이 단위 테스트로 검증되지 않음.
- CSI frame은 기존 신뢰된 로컬 Rust WS 입력. packet origin 인증을 새로 구현한 것은 아님.
- 모델 재학습, smoothing, persistence, Rust/firmware 수정은 이번 범위가 아님.
