# GABES 저장소 분리·통합 — 작업 허브

새 세션 읽는 순서: 이 파일 → **현재 상태** → 해당 단계 → 위험·미확인.
범위: 무거운 결과물·GC 기록을 비공개 아카이브로 분리. 경량 main에서 개발 그룹과 UI 그룹을 각각 통합.

## 현재 상태

- 2026-11-09: S0 결정 완료 (R1–R8). 파일 이동·코드 수정 없음.
- 2026-11-09: S1 진행 중.
  - 미커밋 분류, Drive 중복 객체 정리, 전체 pytest 1회 실행 완료. 결과는 아래 **S1 진행 기록**.
  - 2026-11-09 사용자 "착지 진행" → 통과 묶음 로컬 커밋 완료. push 없음.

    | 커밋 | 묶음 |
    |---|---|
    | 9e6a2a1 | C1 .gitignore |
    | bd30167 | C2 Rydberg 11-04 재최적화 |
    | 5780ce4 | C4a Rydberg 공용 함수 연결 |
    | 65c35f3 | C4b SAS 값 변경 |
    | c3db9ca | C5 PSR 테스트 |
    | b3863db | C6 UI 노트·스킬 |
    | f7301f2 | C7 출판 문서 |
    | 456b33f | C8 GC 기록 |
    | 82e90da | C9 squeezing |
    | 04cff7d | C10 daily report |
    | (이 커밋) | C11 README·checklist·허브 |

  - **C3 로컬 착지.** 사용자 S1 동결 해제 및 C3 범위 확대. FWM 보정 4종, [문헌 DB](../../references/fwm_experiments/README.md), 10-23·10-30 분석 기록, [FWM 개발 로그](../../analysis/fwm_lite/DEVLOG.md), 잡음 라벨을 반영한 [User Guide 원본](../Userguide/GABES_User_Guide_v2.html)·[배포본](../../static/GABES_User_Guide.html)을 포함.
  - **⟦AI⟧ C3 검증:** 기존 실패 10/10 통과. FWM·UI·문서·기존 수치 기준선 742건과 Stark·Zeeman 검증 10건 통과(54개 파일). [과거/현재 AST 대조](../../analysis/fwm_stark_calibration_20261023/source_drift_c3.json). DB 원문 검증 18건 통과. 다운로드한 원문은 기존 정책대로 로컬 보관.
  - Magneto 회귀 2건은 별도 R2 커밋 `2460c6c`. 분리 태그와 전체 기준선 검증은 대기.
  - **깨끗한 checkout 확인 (8989a09):** 착지 영향 테스트 12개 파일 → 235 통과 / 1 실패.
    - 실패: `tests/quantum/test_variance_reduction_pilot_shadow.py::test_preregistration_does_not_touch_future_grid`.
    - 착지와는 무관함. `variance_reduction_2026_10_13/review.json`(488건)과 `pilot-shadow-review.json`(248건)의 `input_byte_ledger` 키가 `G:\내 드라이브\Coding\GABES …` **절대경로**라서, 원래 위치에서만 통과함.
    - **분리 위험:** 아카이브를 다른 경로에 clone하면 이 검증기가 깨짐. S4 전에 경로를 상대경로로 바꿔 읽는 방식이 필요함. 봉인 기록은 수정하지 않음.
  - stale `.git/index.lock`(11-07 05:01, 0 B, git 프로세스 없음)을 지움.

## S1 진행 기록 (2026-11-09)

사용자 결정 (2026-11-09):
- 전체 pytest 통과 후 묶음별로 로컬 커밋. 실패와 연결된 묶음은 보류하고 원인을 보고.
- push는 아카이브를 비공개로 바꾼 뒤에만.
- `analysis/squeezing/operating_point_20261014/coarse.jsonl`(149,474,111 B, 100 MB 초과)은 분리 후 아카이브 Release 첨부. 저장소에는 `coarse.jsonl.external.json`(sha256)만 둠.
- `analysis/rydberg_gold_optimization_20261104/`는 아카이브용으로 커밋.

⟦AI⟧ 실행·확인:
- **정리:** `.git/objects`의 Drive 충돌 사본 `(1)` 87개를 지움. 모두 원본과 바이트 동일.
- **`.gitignore`에 추가:** `/.playwright-cli/`, `mpl_cache/`, `numba_cache/`, `__pycache__/`, 위 coarse.jsonl.
- **전체 pytest:** 현재 작업트리 그대로 `python -m pytest -q --durations=20` → **12 실패 / 3,721 통과 / 3 skip, 1,182 s**.
  - **FWM 묶음 관련 10개** (깨끗한 HEAD에서는 통과 → 미커밋 작업이 원인):
    - 라벨 변경 3개: `test_fwm_excess_noise` 2개, `test_fwm_generic::test_full_scan_plot_uses_display_solver_detail`. 기대값 `Squeezing [dB]` ≠ 새 라벨 `Intensity-difference noise [dB]` / `Gain-referred noise diagnostic [dB]`.
    - rail 선택 4개: chi/stark/zeeman integration. `fwm__gain_calibration_model` selectbox에 MODEL_CHI/STARK/ZEEMAN 옵션이 없음.
    - 기록 receipt 3개: stark validation 1개, zeeman stage1/stage2. 기록된 소스 sha256 ≠ 현재 소스.
  - **magneto 2개는 HEAD 기존 실패:** `test_magneto_fast_solver::...frozen_scheme[optical_linear]`(`rotation_asd_rad_per_sqrt_hz` KeyError), `test_magneto_noise::test_noise_controls_render_toggle_and_advanced_settings`(`magneto__show_shot_noise` 위젯 없음). 6e36a17에서는 통과, **7896f59(linked spectrum workspaces, 푸시됨)부터 실패**.
- **TPD-angle staged 삭제는 의도 없음.**
  - 10-01 기록(`analysis/fwm_gain_hotfix/tpd_gain_diagnostic_20261001/source_manifest_before.json`)에도 같은 staged 목록이 있음. 그 이전부터 남은 오래된 index임.
  - 작업 파일은 HEAD와 바이트 동일하고, v5/v6 `*_loss0_tpd_angle_detail.py`가 이 파일을 import함 → 유지, unstage.
  - staged blob 중 3개(README, `resonant_hanle_squeezing_reference.py`, `scan_variable_audit.py`)는 히스토리에 없음 → unstage 전에 patch로 아카이브에 보존.
  - 스캐너의 실제 작업트리 변경은 09-14 극점 재사용(`response_method`)임. checklist는 done으로 기록했지만 미커밋 상태였음.
- **Rydberg 11-04 재최적화 소스 해시** (`plan.json` `source_sha256` 20개):
  - 현재 디스크와 일치 11개. 이 중 `gabes/observables.py`는 줄바꿈이 섞여 있음(CRLF 707·LF 44) → git checkout으로는 재현 불가, 지금 디스크 바이트만 일치. `evaluator.py`, `run_optimization.py`는 미추적 디스크본.
  - 불일치 9개 중 8개는 커밋 blob에서 바이트 복원 가능: constants·core·schemes/base·schemes/rydberg는 CRLF 변환, beam·species·rydberg_electrometry·rydberg_experiment는 LF 그대로.
  - **`gabes/lineshape.py`는 git 객체 9,189개 어디에도 없음 → 바이트 복원 불가.**
  - `schemes/base.py` drift는 이미 `source_drift.json`·`provenance/base_frozen.py`에 비수치 변경으로 기록돼 있음.

## ⟦AI⟧ Codex 대조 반영 (2026-11-09)

Codex가 GABES 채팅과 git 상태를 대조해 보낸 메시지를 확인·반영함. 기준점 `87aabbc` 일치. 87c977a(모델)·cf069bf(11-03 검증 자료) 확인. 소스 해시 9/20 불일치 확인.

착지 묶음 (확정본). 모두 로컬 커밋만, push 없음.

| 순서 | 묶음 | 조건 |
|---|---|---|
| C1 | `.gitignore` | — |
| C2 | Rydberg 11-04 재최적화 | `.gitattributes`에 `analysis/rydberg_gold_optimization_20261104/** -text`. 복원 가능한 소스 19개를 `provenance/frozen_sources/`에 바이트 그대로 보관하고 `provenance/source_restoration.json`에 복원 범위 기록. lineshape는 복원 불가로 표기. 현 엔진 재실행은 별도 폴더. Gold 절대 실험 교정은 미완료 유지. |
| C3 | FWM 다중 논문 보정 4종 | **로컬 착지.** 실패 10건 해결. README, 문헌 DB, 날짜별 분석, 개발 로그, User Guide 원본·배포본 포함. 실험용 상태와 Kim 음수 OPD gain 재현 실패 기록 유지. |
| C4a | cold-spot: vapor_cell 비율 우선 계산 + Rydberg 공용 함수 연결 + `test_vapor_cell` | Rydberg 결과가 마지막 자리에서 바뀔 수 있음(`n·Tc/Te` → `n·(Tc/Te)`). 상대 1e-12 이하이고 표시값이 동일하면 착지, 차이는 커밋에 기록. |
| C4b | cold-spot: SAS 값 변경 + `test_sas` | 11-04 사용자 결정의 값 단계. 분리 온도에서의 SAS 변화량을 커밋에 정량화하고 checklist 항목 갱신. |
| C5 | PSR 테스트 2개 | — |
| C6 | User Guide, `docs/ui_redesign`, weekly-review 스킬, `.impeccable` | — |
| C7 | 출판 문서: quotient v4→v3.1/v3.2, analytic v3, squeezing v8 + v7 과학 자료, `test_docs_consistency`, `.gitattributes` | — |
| C8 | GC 문서·증거(미추적 467 MB 포함), GC 스크립트 3개·`tests/quantum` 3개 | 바이트 보존(`-text`) |
| C9 | squeezing 스캐너 극점 재사용, `analysis/pole_reuse`, operating_point 폴더(coarse.jsonl 제외) | 오래된 index를 unstage하고, 히스토리에 없는 staged 3개는 patch로 보존 |
| C10 | daily report | 자동 작성이 계속됨: 착지 시점까지만 포함 |
| C11 | README(FWM 문단 제외), checklist, 이 허브 | — |

- 태그·기준선: C1–C11 착지와 magneto 회귀 해결 뒤 `pre-split-<date>` 태그. 그 상태로 전체 pytest를 다시 돌려 기준선으로 저장.
- S2 FWM 테스트 분할:
  - **main:** chi_gain, chi_integration, literature_gain, stark_gain, stark_integration, stark_response, zeeman_gain, zeeman_integration, zeeman_response, gain_calibration_integration (10개).
  - **아카이브:** chi_validation, literature_validation, stark_validation, zeeman_validation, stark_cache, zeeman_cache (6개, 날짜별 기록을 직접 읽음).
  - cache 테스트가 읽는 기록을 런타임도 쓴다면, 그 기록은 `gabes/data/`로 옮김.

재개 지도 (Codex 제안 채택):

| 시점 | 위치 | 작업 |
|---|---|---|
| S1–S2 (분리 전) | 현재 저장소 | cold-spot 두 커밋, TPD-angle 정리, main이 아카이브 없이 실행·회귀 가능하도록 의존 정리 |
| 분리·패키지 정리 후 | `gabes/` | Rydberg 물리: 빔/Rabi → 절대 투과율 → 광학 운전점 → 잡음 교정 → 희소 격자·LO 예측선 재검증 |
| 분리·패키지 정리 후 | `gabes_lab/pipelines` | 가열·최적화 도구, squeezing 스캐너 |
| 분리·패키지 정리 후 | `gabes/` | FWM 다음 후보: 유한 seed 포함 두 색 원자 응답. 논문·장치 분리 검증과 상수 기준선 유지 |
| UI 통합 후 | `gabes/` | PSR S5–S6 |
| UI 통합 후 | `gabes_ui/` | Magneto 지도·통합 검수(UI 그림 연구 4–5단계), SABES 디자인 후보 |
| 새 저장소 재연결 후 | Streamlit Cloud | 공개 앱 반영 확인 |
| 분리 태그 | 아카이브 | 봉인 GC 증거를 원래 경로·해시로 검증 |

- 이미 푸시된 것: FWM 위상 스캔(afb31f6), PSR S0–S4(653e63a), SABES 테이블 마감(87aabbc).
- 단 7896f59가 magneto 테스트 2개를 깨뜨림 → 소유 채팅에서 수정 필요.

## 사용자 요청 (이 작업 패키지 대화, 요지)

- 두 그룹을 각자 하나로: [tests + tools + analysis], [gabes_ui + assets]. SABES의 UI 부분도 UI 그룹으로. 디자인 스타일·스킬을 통합해서 관리.
- UI 경로 상수·로더 통일. 재사용 모듈(블록) 작성.
- 통합 구조와 블록 사용법을 AGENTS.md에 넣어 에이전트가 매번 새로 짜지 않게. 철칙 아님. 필요한 분석이 기존 구조에 없으면 새로 만들어도 됨.
- 통합 이전 코드는 archive로 보내거나 새 통합 폴더에 정리. 원본↔새 경로 매핑과 이전 사유를 Migration Manifest로 남겨 GC가 계보 추적.
- 이전 후 GC·GABES 코드 테스트.
- .gitignore 재검토. 공유받는 사람에게 불필요한 무거운 analysis 결과·docs 결과물은 main에서 제외. main은 분석용 블록 도구 위주로 가볍게. GC 결과 추적은 별도 아카이브 프로젝트.

## 사용자 결정 (2026-11-09)

8건 모두 질문 선택지 중 ⟦AI⟧ 추천안 선택.

| ID | 결정 | 내용 |
|---|---|---|
| R1 | 아카이브 비공개 | 현재 `Shake2313/fwm-squeezing-app`을 이름 변경 → 비공개 아카이브. 전체 히스토리·경로·해시 보존. 이름 `GABES-archive`는 ⟦AI⟧ 제안, 미확정. |
| R2 | main 히스토리: filter-repo | main에 남는 경로의 이력만 추출. 커밋 해시 새로 생김. 옛↔새 commit-map은 Split Manifest에 기록. |
| R3 | main 이름 `GABES` | 새 저장소. `fwm-squeezing-app` 이름 재사용 금지 (옛 이름 → 아카이브 redirect 유지). |
| R4 | 과학 문서는 최신본만 main | main `docs/papers/`에 세 문서 최신 PDF/HTML. tex 원본·옛 판본·개발 기록은 아카이브. |
| R5 | UI 패키지 `gabes_ui` 유지 | `assets/` → `gabes_ui/assets/`. SABES 화면 코드(`sabes/components/`, `sabes/layout/render.py`, `sabes_page.py`) → `gabes_ui/sabes/`. 루트 고정: `streamlit_app.py`, `.streamlit/`, `static/`. |
| R6 | 개발 패키지 `gabes_lab` | `gabes_lab/blocks` + `gabes_lab/pipelines`. `tools/` 흡수. `tests/`는 루트. |
| R7 | 테스트는 pytest 전용 | 루트 pytest 설정/`conftest.py`가 import 경로 해결. 파일별 `sys.path.insert` 제거. 단일 실행 `python -m pytest tests/test_x.py`. README 안내 갱신. |
| R8 | GC 규칙 개정 | 적용 시점 S3–S4, 아카이브의 `docs/grand_challenge/AGENTS.md`에. 문구 아래. |

R8 문구. 개정 방향은 사용자 승인, 문구는 ⟦AI⟧ 초안. `Preserve historical evidence bytes and paths` 줄을 교체:

> Preserve historical evidence bytes and paths inside the archive repository: reports, cache records, declarations, source ZIPs, archived controllers, figures, logs, validators. Evidence recorded before the repository split is re-verified at the split tag, where its original paths and engine exist. After the split the archive pins the engine as a submodule; archive HEAD need not run pre-split validators unchanged. New evidence records provenance as {repo, commit, path}. New results need distinct outputs; failed gates stay recorded.

## ⟦AI⟧ 설계 — 2026-11-09 사용자 "이렇게 하고"로 방향 승인, 세부는 실행 중 조정

### 측정 기준선 (2026-11-09, HEAD `87aabbc`)

- 원격: 공개, GitHub diskUsage 375,174 KB, 커밋 189개. 원격 브랜치 `legacy-main`, `generalize-obe-schemes`, `claude/*` 2개.
- HEAD 추적: 2,759 파일, 1.82 GB.
  - `docs/grand_challenge` ≈ 1.63 GB (thermal_campaign_v2 984 MB, v1 372 MB, rb_thermal_path_cache_v1 100 MB)
  - `docs/daily_report` 40 MB, `analysis` 110 MB
  - `.py` 전체 6.9 MB, JSON 1,639 MB
- 히스토리 blob 2.2 GB raw (620 MB zlib). main에 남길 경로만: blob 1,405개, 53.6 MB raw / 21.3 MB zlib.
- origin/main에서 도달 가능한 최대 blob 32.5 MB.
- 커밋 해시 참조: json 26, md 22, checklist 12.
- 저장소 URL 참조: 38 파일 73회. README, User Guide, Rydberg brief v3 tex(clone URL), daily report의 issue 링크 27회 등.
- 로컬 `.git`: loose 624 MiB, pack 0, Google Drive 충돌 사본 `(1)` 89개.

### 경계

| GABES main (공개) | GABES-archive (비공개) |
|---|---|
| `gabes/`, `gabes_ui/` (+ assets, SABES 화면), `sabes/` 모델 | `docs/grand_challenge/` 전체 |
| `gabes_lab/` 블록·재사용 파이프라인 (`rydberg_cell_heating` workflow, squeezing 스캐너 등) | GC 코드: `analysis/grand_challenge/`, `tests/quantum/`, `tools/thermal_campaign_*` |
| GC 외 `tests/`, `tools/ui_audit`, `benchmark_psr` → `gabes_lab` | 날짜별 `analysis/*` 기록, `output/` |
| 모델 입력 데이터 `gabes/data/`, `references/*_experiments` (2.1 MB) | `docs/daily_report/`, 보고서 옛 판본·tex, SABES·Rydberg·Hanle 연구 문서 |
| README, AGENTS.md, User Guide, `docs/ui_redesign/`, `docs/repo_restructure/`, `static/` | GC 스킬 `grand-challenge-head-manager`, GC 규칙, publication policy와 판본 이력 |
| `docs/papers/` 세 과학 문서 최신본 (R4) | 세 문서의 판본 이력·개발 로그 |

- 아카이브는 main을 submodule(`engine/`)로 커밋 고정. 아카이브 pytest 설정 `pythonpath = engine`.
- 분리 태그: 아카이브에 `split-<date>`. 옛 GC 증거 재검증은 이 태그에서. 원래 경로가 살아 있어서 경로 번역 불필요.
- 새 GC 증거: `source_provenance` 스키마 v2, 출처 = `{repo, commit, path}`.
- 이전 계획의 `gabes_lab/records/`, `gabes_lab/grand_challenge/`는 분리로 대체. 날짜별 기록과 GC 코드는 아카이브 원래 경로에 그대로 둠.

### 분리 전에 끊을 main → 결과물 의존 (S2, 현재 저장소에서)

- 런타임 출처 인용 문자열 4곳 → `GABES-archive@<commit>:<path>` 형식:
  - `gabes/fwm_experiment_gain.py`
  - `gabes/fwm_gain_closure.py`
  - `gabes/fwm_measured_gain.py`
  - `sabes/apparatus_20261016.json`
- 모델 입력 데이터 → `gabes/data/`:
  - `analysis/fwm_experiment_20261022/gain_profile.json`
  - `docs/grand_challenge/gain_calibration_2026_10_16/interpolation_record.json` (+ fit, spline_fit)
- 결과물을 읽는 테스트 약 16개. 테스트마다 셋 중 하나:
  - 최소 fixture를 `tests/fixtures/`로 복사
  - 해당 코드를 `gabes_lab/pipelines`로 승격
  - 테스트째 아카이브로
- 개별 처리:
  - `tests/test_psr_spectrum.py`가 `analysis.grand_challenge.reference.atomic_qrt`를 import함 → `atomic_qrt`를 gabes로 승격하거나 테스트를 아카이브로.
  - `test_docs_consistency`, `test_fwm_docs_consistency` → main 문서용과 아카이브 문서용으로 분할.
- 완료 조건: 현재 저장소 full pytest 통과. 그래야 S3가 순수한 파일 나누기가 됨.

### main .gitignore·크기 규칙

- 결과물은 기본 비추적.
  - 파이프라인 출력은 `runs/`·`output/`·`tmp/` (ignore).
  - `*.jsonl`, `*.npz`, `*.log`, `*.png`도 기본 ignore. 허용 목록 `gabes/data`, `tests/fixtures`, `gabes_ui/assets`, `static`만 예외.
- 크기 감시 테스트: 허용 목록 밖에서 추적되는 파일이 1–2 MB를 넘으면 실패. 문턱값은 S2에서 확정.
- 결과를 아카이브로 보내는 건 명시적으로만. `gabes_lab` 블록 `publish_to_archive`가 main 커밋과 sha256이 담긴 manifest를 함께 복사.
- 정리 대상:
  - 추적 중인 `output/` 52개 untrack
  - `.claude/skills/` 추적 (`settings.local.json`은 계속 ignore)
  - `/.playwright-cli/` ignore

### 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| S0 | 결정 R1–R8 | 완료 2026-11-09 |
| S1 | 동결: 다른 세션 작업 착지. untracked analysis 1,499개 분류(아카이브 커밋/삭제). `(1)` 사본 정리. 태그 `pre-split-<date>`. 기준선 저장: full pytest, `tests/quantum`, GC 검증기. | 진행 중 (위 S1 진행 기록) |
| S2 | 위 의존 끊기, .gitignore·크기 감시, `output/` untrack → full pytest | 대기 |
| S3 | GitHub 이름 변경 → 아카이브 비공개. `git filter-repo`로 main 경로 추출 → 새 `GABES` push. Split Manifest(경로 목록, commit-map) 생성. Streamlit Cloud 재연결. README·User Guide URL 갱신. | 대기 |
| S4 | 아카이브: main 경로 제거, submodule `engine/`, R8 적용. 분리 태그에서 `tests/quantum`·GC 검증기 재실행. | 대기 |
| S5 | main 통합 (이전 P2–P5) | 대기 |
| S6 | 검증: main full pytest, 앱 기동·SABES 렌더, User Guide 빌드, md 링크, 스킬·메모리 옛 경로 갱신 | 대기 |

S5 세부:

1. **UI 파일럿.** R5 이동, 경로 상수·로더 단일화, `gabes_ui/DESIGN.md` 허브, 프로젝트 스킬 `gabes-ui-design`.
2. **개발 그룹.**
   - `gabes_lab` 생성, `paths` 블록(위로 올라가며 표식 파일로 루트 탐색), 루트 conftest.
   - `git mv` 후 경로 치환.
   - 이동과 내용 수정은 별도 커밋 (`git log --follow` 유지).
3. **블록.** provenance, io, figures, snapshot, runner, publish_to_archive. 블록에 물리 금지 (물리는 `gabes`).
4. **규칙.** AGENTS.md에 5–8줄. 폴더 지도, "블록 먼저 찾기 → import → 없으면 새로 작성, 범용이면 블록에 추가", 철칙 아님. 세부는 `gabes_lab/README.md`.

Manifest 두 종류. 둘 다 스크립트로 생성하고, 손으로 쓰지 않음.

- **Split Manifest (S3).** 저장소 단위.
  - 내용: main/아카이브 경로 목록, filter-repo commit-map, 분리 태그, 각 쪽 HEAD.
  - 위치: 양쪽에 같은 내용.
- **Migration Manifest (S5).** main 내부 이동.
  - 내용: `old_path`, `new_path`, 이동 전·후 sha256, 변경 종류(이동만 / 이동+경로 수정 / 아카이브 / 삭제), 사유, 기준 커밋.

### 위험·미확인

- **Streamlit Cloud.** 새 저장소로 재연결한 뒤 앱 URL이 유지되는지 미확인. S3 전에 확인.
- **옛 URL은 외부인에게 404.** 비공개 아카이브 + 이름 변경이라 옛 URL은 아카이브로 redirect되지만, 외부인은 접근 불가.
  - 이미 전달된 Rydberg brief v3의 clone URL, 옛 User Guide·README 사본이 해당.
  - main 판 문서는 S3에서 새 URL로 갱신. 이미 전달된 문서는 수신자에게 새 URL 안내 필요(사용자 몫).
- **issue.** 옛 issue는 아카이브에 남음. daily report 링크와 일치.
- **크기 추정.** filter-repo 결과 크기는 blob 집계 기반 추정치.
- **이미 공개된 내용.** 기존 fork·clone에서 회수 불가.
- **G: Drive 위 `.git`.** loose 624 MiB. gc·대량 이동 때는 Drive 동기화 일시정지를 고려. stale `.git` lock 주의.
