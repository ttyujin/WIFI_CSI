# 연구 서비스의 안전한 정리 기록

분석일: 2026-10-01. 기준은 현재 로컬 checkout과 사용자 지정 실행 경로다.
목표는 현재 기능 보존 및 원본 RuView와 연구 확장 부분의 구분이다.

## 수행한 정리

- 루트 README 첫 부분에 연구 서비스의 목적, 실행 흐름, 역할 문서 링크를 추가했다.
- 원본 README 본문은 보존하고 원본 설명임을 명시했다.
- `RESEARCH_SERVICE.md`에 모델·포트·환경변수·실행 순서·간접 의존성을 기록했다.
- 삭제: **없음**. 검토 후보 중 사용자 지정 삭제 조건 전체를 만족한다고
  입증할 수 있는 파일을 확정하지 못했으므로 기능 보존을 우선했다.
- runtime 코드, 모델, 특징, 임계값, API, UI, firmware 및 포트는 변경하지 않았다.

## 확인한 근거

| 검사 | 확인 결과 |
|---|---|
| Git 상태 | 기존 `.gitignore`, Rust `main.rs` 수정 및 untracked 연구 코드가 있었음. 사용자 변경은 보존 |
| Python AST/import 및 파일명/함수 참조 | 현재 backend의 `extract_normalized_features`가 CV → baseline → preprocessing 모듈에 연결됨 |
| frontend import/asset 참조 | components, hooks, API, utilities, CSS 및 4개 GIF가 앱/테스트/빌드에 연결됨 |
| npm/Vite | package scripts, tsconfig, test setup, lockfile, 0.0.0.0/8090 및 hostname:8010 확인 |
| Rust module/router | `main.rs`, `lib.rs`의 모듈, `/ws/activity/csi`, 수신→recordable predicate→broadcast→WS 확인 |
| Cargo metadata/dependency | sensing-server lib/bin, examples/tests/bench 및 로컬 path dependency 확인 |
| build/CI/설정 | workspace manifests, toolchain, submodules, 원본 UI 정적 경로, Python CI/Docker 및 테스트 참조 확인 |
| 문서/실행/model/data | README, ADR-018/295/299, 모델 경로, profile 저장 경로, 연구 recording/processed 경로 확인 |
| 선택적 안내 도구 | 로컬 `harness/ruview/bin/cli.js guidance`의 소스 확인 기능 사용. Ruflo MCP는 없어 로컬 소스 검색으로 대체 |

참조 검사에는 `rg`의 파일명·함수명·import 검색, Cargo metadata, 각 진입점과
처리/라우트의 직접 읽기를 함께 사용했다. 검색 결과 없음만으로 삭제를 승인하지 않았다.
현재 적용 범위의 의존성을 확인한 것이며, 저장소 전체의 모든 동적 경로가
완전히 불필요하다는 증명은 아니다.

## 유지한 코드 분류

| 분류 | 경로/예시 | 보존 이유 |
|---|---|---|
| A: 직접 필요 | firmware, Rust `main.rs`/`activity_recording.rs`, Python backend/alerts, 모델, React 앱 | 현재 CSI→추론→API→웹→메일 실행 경로 |
| C: 간접 필요 | `validate_heightLayoutA_cv.py`, `train_activity_baseline.py`, `prepare_activity_dataset.py` | 현재 추론의 함수/모듈 import. 예전 실험 파일명이어도 runtime 의존성 |
| C: 빌드/테스트 | Rust signal/physics/engine/worldgraph/vitals/pose/bridges, Cargo/lockfile/toolchain, tests, npm 설정 | 같은 바이너리 또는 테스트·Cargo dependency에 연결 |
| B/C: 원본 UI | `ui/`의 Senior Monitor 외 코드 | 기존 실행 옵션 `--ui-path ..\ui`, 정적 서빙 및 원본 JS 회귀 테스트 |
| B/C: 연구 재현 | `evaluate_*`, `validate_*`, `compare_*`, `inspect_*`, artifact/CLI inference/QC/collection 도구 | 현재 서비스 직접 진입점은 아니지만 모델/데이터 파이프라인 관련 |
| B/C: 원본 인프라 | `archive/v1`, `docker`, `scripts`, `examples`, `assets`, `vendor`, `harness`, 원본 문서 | CI·Docker·submodule·example/테스트·문서 참조 또는 간접 필요성 배제 불가 |
| B: 사용자 백업 | `v2/crates/wifi-densepose-sensing-server/src/main.rs.before_activity_ws` | Cargo 대상은 아니고 직접 파일명 참조도 확인되지 않았지만, 사용자 untracked 백업이며 Rust 변경 복구에 필요할 수 있음 |

D/E 삭제 확정 파일은 없다. 개인 데이터·recording·processed CSV·모델·로컬
실행 파일·기존 빌드 캐시는 삭제하거나 이름을 변경하지 않았다.

## Clone 시 이해해야 할 점

루트 README는 연구 서비스로 안내하고, 원본 설명은 뒤에 별도로 유지한다.
새 개발자는 직접 runtime, 간접 특징 추출 의존성, 연구 도구, 원본 빌드/테스트
인프라를 구분할 수 있다. 모델은 `.gitignore`의 `models/`에 의해 제외된다.
GitHub에 실제 연구 변경분이 commit되어 있어야 clone에 포함되며, 모델과
운영 credential은 별도로 준비해야 한다. 이 작업에서는 commit/push하지 않았다.

## 검증 결과

| 검증 | 결과 |
|---|---|
| backend/profile/email Python tests | 34개 통과. SMTP/CSI는 mock/fixture |
| dataset/QC/bimodality/baseline/artifact Python tests | 39개 통과 |
| PowerShell collection helper | 기존 15개 assertion 통과. 실제 recording은 하지 않음 |
| 보호자 웹 `npm test` | 4개 파일, 39개 테스트 통과 |
| 원본 UI/desktop Node 회귀 테스트 | 29개 통과 |
| 보호자 웹 `npm run build` | TypeScript 검사와 Vite production build 성공 |
| Python 문법/의존성 | runtime과 특징 추출 의존 모듈 5개 `py_compile` 성공, 로컬 도구 30개 AST 파싱 성공, 검증 환경 `pip check` 성공 |
| 현재 모델 read-only load | `RandomForestClassifier`, LYING/MOVING/SITTING, 918차원 확인. 재학습/저장하지 않음 |
| Rust sensing-server | `cargo check --locked --offline` 성공. 해당 package의 전체 `cargo test`: 859개 통과, 2개 ignored, 0개 실패. 기존 경고는 유지 |
| 보호 대상 무결성 | 소스·설정·GIF·모델·기존 실행 파일·백업 1,370개 SHA-256 전후 동일 |
| 문서 | 추가 안내의 로컬 링크 6개 유효, `git diff --check` 성공 |

Rust 검증은 `v2/target/activity-cleanup-check`를 사용했다. 현재 운영 중인
`target/activity-live-v2-build/release/sensing-server.exe`는 교체하지 않았다.
Python 의존성은 별도 `v2/target/activity-cleanup-python` 환경에 설치했으며
운영 Python이나 requirements/lockfile은 변경하지 않았다. 이 두 경로 및
frontend `dist`는 검증 과정에서 생성된 로컬 산출물이고 Git에 추가하지 않았다.

첫 Python 연구 테스트 시도는 실행 디렉터리/미설치 의존성 문제로 실패했지만,
올바른 `v2` 디렉터리와 별도 환경에서 재실행하여 39개 모두 통과했다.
첫 Vitest 시도는 worker 시작 timeout이 있었으며, 소스나 test 설정을 수정하지
않고 실행 권한을 허용해 재실행했을 때 39개 모두 통과했다.

기능 코드 변경 및 파일 삭제는 없다. 위 테스트 범위는 현재 연구 Python 도구,
보호자 웹, 관련 원본 UI와 sensing-server package이며, 저장소 전체의 모든
workspace crate/원본 Python 환경/선택적 외부 인프라 테스트를 실행했다는 뜻은 아니다.
로컬 8010/3000 API 연결은 실패하여 실제 ESP32 연속 수신·휴대폰 접속·Gmail 수신은
이번 작업에서 재검증하지 않았다. 자동 테스트를 실제 센서·SMTP 전달의 검증과
구분해야 하며, 이 결과로 모든 운영 환경에서 오류가 없다고 보장하지 않는다.
