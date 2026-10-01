# 보호자 이메일 알림 / 프로필 wizard

## 구조와 보존 범위

기존 Python StreamState가 정상 prediction을 확정한 직후, lock 밖에서 알림
조건을 검사합니다. UI polling이나 브라우저 timer는 발송에 관여하지 않습니다.

정상 prediction → LIVE interval/확인된 duration → interval·stage 중복 guard → bounded Queue(64)
→ 전용 email-alert-worker → SMTP_SSL(smtp.gmail.com:465, 인증서 검증, timeout10초).

Rust/UDP/firmware, raw306 CSI,3초 window/1.5초 hop,918 feature extraction,
RandomForest joblib와 연구 데이터/분석 코드는 변경하지 않았습니다.
서비스는 MOVING/STAYING이며 P(MOVING)>=0.70 기준을 유지합니다.

LAN 설정: API_HOST=0.0.0.0, Vite host=0.0.0.0/port8090,
API_BASE=http://현재 페이지 hostname:8010.
PC는 localhost:8090, 같은 Wi-Fi의 스마트폰은 PC LAN IP:8090을 사용합니다.
새 profile API는 동일 hostname의8090 Origin만 허용합니다(CORS/CSRF 방어).
인증 시스템은 추가하지 않았으므로 신뢰된 LAN에서만 사용하고 외부로 port-forward하지 마세요.

## Profile

저장 경로: C:\Users\user\Desktop\wifi\RuView\v2\data\senior-monitor\profile.json

이 파일은 실제 저장 요청 때만 생성됩니다. DB 없음. 임시 파일에 기록/fsync 후
같은 디렉터리 내 atomic replace. 실패하면 기존 파일/메모리 profile을 보존합니다.
전용 profile lock은 activity lock과 분리됩니다. 전체 로컬 디렉터리를 .gitignore 처리합니다.
저장되는 정보는 name,gender,guardian_email 세 필드뿐이며 비밀번호는 절대 넣지 않습니다.

GET /api/profile:

```json
{"profile": null, "email_alerts_enabled": false}
```

POST /api/profile (Content-Type: application/json):

```json
{"name": "홍길동", "gender": "female", "guardian_email": "guardian@example.com"}
```

성공200: success=true, profile, email_alerts_enabled.
잘못된 이름/성별/이메일/추가 필드/JSON은400, disk 저장 실패는503,
허용되지 않은 Origin은403. 요청 최대4096bytes.
이름 trim1~40자/제어문자 금지, gender female|male, 기본 ASCII email 문법/254자 제한.
이메일 소유 여부를 인증하는 기능은 없습니다.

최초 UI는 이름→성별 카드(여성=grandma, 남성=grandpa)→이메일 순서입니다.
마지막 사용자 클릭에서만 POST하며 backend 저장 성공 후 main으로 갑니다.
저장 중 중복 클릭을 막고 실패하면 같은 단계에서 재시도합니다.
기존 localStorage name/characterType은 삭제하지 않고 이메일 없는 경우3단계부터 진행.
프로필 수정은 세 필드를 모두 변경할 수 있습니다. GET의 backend 값이 캐시보다 우선합니다.
캐시 쓰기 실패 시 서버 성공을 되돌리지 않고 명확히 경고합니다.

## 알림 조건과 중복 방지

기본 threshold는 STAYING 상태 확인10800초(3시간), MOVING 상태 확인3600초(1시간),
공통 주의14400초(4시간), 공통 위험21600초(6시간)입니다.
이는 변경 가능한 서비스용 heuristic 기본값이지 임상/논문에서 검증된 기준이 아닙니다.
운영 threshold는 실제 실험 후 결정해야 합니다.

환경변수:

- WIFI_ELDER_STAYING_NOTICE_SECONDS (기본10800)
- WIFI_ELDER_MOVING_NOTICE_SECONDS (기본3600)
- WIFI_ELDER_CAUTION_SECONDS (기본14400)
- WIFI_ELDER_DANGER_SECONDS (기본21600)
- WIFI_ELDER_SENDER_EMAIL (기본 wifieldersender@gmail.com)
- WIFI_ELDER_SENDER_APP_PASSWORD (기본값 없음)

기존 실행 명령과의 호환을 위해 `WIFI_ELDER_STAYING_ALERT_SECONDS`와
`WIFI_ELDER_MOVING_ALERT_SECONDS`도 새 NOTICE 변수가 없을 때 fallback으로 읽습니다.
새 설정에는 NOTICE 이름을 사용하세요.

시작 시 한 번 읽습니다. 네 단계 값은 유한한 양수이며 각
NOTICE < CAUTION < DANGER 순서여야 합니다. credential이 없거나
설정이 잘못되면 한 번 warning을 출력하고 email만 disabled, CSI/API는 계속 동작합니다.
프로필이 유효해야 email_alerts_enabled=true가 됩니다.
이 값은 설정 가능 여부이지 Gmail 인증/실제 수신 성공을 확인했다는 의미가 아닙니다.

각 정상 interval에 증가하는 interval_id를 부여합니다.
정상 LIVE에서 confirmed_duration_seconds >= 해당 threshold일 때 stage를 한 번만 claim합니다.
중복 키는 `(interval_id, NOTICE|CAUTION|DANGER)`입니다. 따라서 같은 interval에서
상태 확인·주의·위험을 각각 최대 한 번 시도하지만 동일 stage는 추가 prediction,
profile 수정 또는 SMTP 실패로 다시 발송하지 않습니다.
상태 전환 또는 disconnect 후 첫 prediction은 새 interval_id이므로 새 알림이 가능합니다.
Queue가 가득 찬 경우도 해당 알림을 skip하며 무한 재시도하지 않습니다.

CONNECTING/RECONNECTING/UNKNOWN/invalid CSI/프로필 없음/threshold 미도달은 발송 없음.
disconnect는 기존대로 마지막 prediction에서 종료되고 gap은 alert duration에 포함되지 않습니다.
복구 socket 연결만으로 LIVE가 되지 않으며 새3초 window의 첫 prediction부터 새 시간이 시작됩니다.
worker는 대기 중 단절/상태 전환/수신자 변경을 발견하면 알림을 취소합니다.
SMTP connect/login 이후 DATA 직전에도 재검사합니다.

실패 정책: SMTP 예외는 worker에서 처리하고 예외 종류만 로그에 남깁니다.
SMTP 응답 상실은 이미 수신되었는지 알 수 없으므로 자동 retry하지 않습니다.
즉 interval·stage당 최대1회 전송 시도이며 수신 보장/exactly-once delivery는 아닙니다.
이미 SMTP DATA 전송에 들어간 메일은 그 순간 CSI가 끊겨도 취소/회수할 수 없습니다.

메일 제목은 `[상태 확인]`, `[주의]`, `[위험]` 단계를 표시하며 본문은
HH:MM:SS/감지 시각을 포함합니다. 주의·위험은 지속시간 기반 서비스 단계이고
건강 상태나 응급 여부를 판단한 결과가 아님을 본문에 명시합니다.

## credential 설정과 실제 Gmail 단계별 수동 검증

자동 테스트는 SMTP를 mock하며 실제 Gmail이나 보호자에게 전송하지 않습니다.
아래는 사용자가 본인 테스트 수신 주소로 직접 실행하는 절차입니다.

Google App Password는2단계 인증이 필요합니다. 계정 정책에 따라 사용 가능 여부가
달라질 수 있습니다. 일반 로그인 비밀번호를 사용하지 마세요.
[Google 공식 App Password 안내](https://support.google.com/accounts/answer/185833?hl=ko)
[Google 공식 SMTP SSL 설정](https://support.google.com/a/answer/176600?hl=ko)

1. 기존 RuView(HTTP3000, WS3001, UDP5005)와 ESP32 정상 수신을 유지합니다.
2. 기존 Python Activity backend가 있으면 Ctrl+C로 종료합니다.
3. 기존 모델 의존성이 설치된 Python 환경의 **같은 PowerShell 창**에서 설정합니다.
   아래 secure prompt는 실제 비밀번호를 명령 기록에 직접 쓰지 않기 위한 것입니다.
   환경변수/프로세스 메모리에는 실행을 위해 평문이 존재하므로 PC 접근을 보호하세요.

```powershell
cd C:\Users\user\Desktop\wifi\RuView\v2
$env:WIFI_ELDER_SENDER_EMAIL = 'wifieldersender@gmail.com'
$mailSecret = Read-Host 'Gmail App Password 입력' -AsSecureString
$env:WIFI_ELDER_SENDER_APP_PASSWORD = ([System.Net.NetworkCredential]::new('', $mailSecret)).Password.Replace(' ', '')
Remove-Variable mailSecret
$env:WIFI_ELDER_STAYING_NOTICE_SECONDS = '20'
$env:WIFI_ELDER_MOVING_NOTICE_SECONDS = '20'
$env:WIFI_ELDER_CAUTION_SECONDS = '40'
$env:WIFI_ELDER_DANGER_SECONDS = '60'
python tools/activity_app_server.py
```

4. 별도 터미널에서 frontend 시작:

```powershell
cd C:\Users\user\Desktop\wifi\RuView\ui\senior-monitor
npm install
npm start
```

5. localhost:8090에서 wizard/프로필 수정으로 **본인이 확인 가능한 테스트 이메일** 저장.
   GET /api/profile의 email_alerts_enabled=true 확인(비밀번호는 반환하지 않음).
6. 실제 STAYING을 한 interval로 유지합니다. 첫 prediction 이후 다음 prediction 기준으로
   20초에 상태 확인,40초에 주의,60초에 위험 메일이 각각 한 번 오는지 확인합니다.
   각 단계 사이에 같은 단계 메일이 반복되지 않아야 합니다. 필요하면 스팸함을 확인합니다.
7. 60초 이후 같은 started_at/interval_id의 STAYING을 계속 유지하여 추가 메일이 없는지 확인합니다.
   MOVING도 별도의 새 interval에서 같은20/40/60초 순서로 검증할 수 있습니다.
   상태가 바뀌면 새 interval이므로 다시 단계별 알림이 가능하다는 점을 구분하세요.
8. 브라우저를 닫아도 backend가 실행 중이면 조건 검사가 계속됩니다.
   별도 추가 메일 검증은 새로운 interval을 의도적으로 만드는 경우에만 수행하세요.
9. SMTP 실패 시 같은 interval의 같은 stage는 재전송하지 않습니다. 인증/네트워크 확인 후
   backend 재시작 또는 새 활동 interval로 별도 검증하세요. 반복 재시작은 피하세요.
10. 검증 즉시 Ctrl+C로 backend를 종료하고 테스트 threshold를 복구합니다.

기본 heuristic으로 복구(검증된 운영 기준이라는 뜻이 아님):

```powershell
Remove-Item Env:WIFI_ELDER_STAYING_NOTICE_SECONDS -ErrorAction SilentlyContinue
Remove-Item Env:WIFI_ELDER_MOVING_NOTICE_SECONDS -ErrorAction SilentlyContinue
Remove-Item Env:WIFI_ELDER_CAUTION_SECONDS -ErrorAction SilentlyContinue
Remove-Item Env:WIFI_ELDER_DANGER_SECONDS -ErrorAction SilentlyContinue
Remove-Item Env:WIFI_ELDER_STAYING_ALERT_SECONDS -ErrorAction SilentlyContinue
Remove-Item Env:WIFI_ELDER_MOVING_ALERT_SECONDS -ErrorAction SilentlyContinue
# 또는 실제 실험으로 정한 초 값을 네 환경변수에 명시적으로 지정하세요.
python tools/activity_app_server.py
```

완전히 알림을 끌 때에는 backend 종료 후 비밀번호 환경변수를 제거하고 재시작:

```powershell
Remove-Item Env:WIFI_ELDER_SENDER_APP_PASSWORD -ErrorAction SilentlyContinue
python tools/activity_app_server.py
```

설정은 해당 PowerShell과 자식 프로세스에만 적용됩니다.
App Password를 소스/.env/JSON/localStorage/스크린샷/로그/README에 넣지 마세요.

## 검증 명령 / 한계

```powershell
# RuView repository root
python -m unittest discover -s v2/tools -p test_activity_app_server.py -v
python -m unittest discover -s v2/tools -p test_senior_monitor_alerts.py -v
python -m py_compile v2/tools/activity_app_server.py v2/tools/senior_monitor_alerts.py
# ui/senior-monitor
npm test
npm run typecheck -- --noUnusedLocals --noUnusedParameters
npm run build
```

backend history/interval/queue/중복 guard는 메모리 기반입니다. 재시작 후 새 interval이며
이전 알림 전송 이력은 복구하지 않습니다. 프로필만 JSON으로 유지합니다.
메일 배달/스팸 처리/Gmail 제한/실제 ESP32 동작은 mock 테스트가 보장하지 않습니다.
단일 PC/단일 profile/신뢰된 LAN MVP입니다. 여러 사용자 인증/공개 인터넷 배포는 범위 밖입니다.
같은8010 port의 중복 backend는 SMTP/inference 시작 전 bind 실패로 차단합니다.
