# GABES 저장소 분리·통합 — 작업 허브

새 세션 읽는 순서: 이 파일 → **현재 상태** → 해당 단계 → 위험·미확인.
범위: 무거운 결과물·GC 기록을 비공개 아카이브로 분리. 경량 main에서 개발 그룹과 UI 그룹을 각각 통합.

## 현재 상태

- 2026-11-09: S0 결정 완료 (R1–R8). 파일 이동·코드 수정 없음.
- 2026-11-09: **S1 완료.** 기준선 확정, 태그 `pre-split-2026-11-09`(이 기록 커밋), push 없음.
  - 사용자 요청: R2 `2460c6c`·C3 `87243e9` 포함 확인 후 S1 전체 테스트로 기준선 확정.
  - ⟦AI⟧ 두 커밋 모두 HEAD 조상 확인. 실행 전후 작업트리 미커밋 0건.
  - ⟦AI⟧ HEAD `87243e9`, 원래 경로에서 `python -m pytest -q --durations=20 -p no:cacheprovider` → **3,733 통과 / 3 skip / 0 실패, 1,297 s**, Python 3.14.2.
    - 기록: [baseline.json](../../analysis/repo_split_s1_20261109/baseline.json), 로그 `pytest_baseline_87243e9.txt`.
    - 포함: `tests/quantum` 전체.
    - 제외: 봉인 캠페인 재실행과 외부 GC 검증기 (AGENTS 범위 규칙).
    - 태그 커밋과 87243e9의 차이는 이 기록 파일·허브·checklist뿐이고, 코드 변경은 없음.
    - 절대경로 GC 테스트는 원래 경로라서 통과함. 다른 경로에 clone하면 실패하므로 S4 전 해결 대상으로 유지.
  - 다음: S2 (main의 결과물 의존 끊기).
- 2026-11-09: **S2 완료.**
  - ⟦AI⟧ main 전용 checkout(36b56bd, 624개 파일): 2,275 통과 / 0 실패.
  - ⟦AI⟧ 원래 checkout 전체(0ad777d): 3,734 통과 / 3 skip / 0 실패, 1,264 s.
  - 기록: [S2 record](../../analysis/repo_split_s2_20261109/README.md).
- 2026-11-09: **S3 준비** (사용자 "준비 작업 시작"). 비파괴 작업만 함.
  - ⟦AI⟧ **filter-repo 리허설** (C:의 bare clone, push 없음):
    - 경로 647개: 규칙 627 + rename 계보 20. 결과는 커밋 195개, **압축 4.60 MiB**(현재 GitHub 375,174 KB 대비 약 80분의 1).
    - HEAD 트리는 0ad777d의 main 쪽 624개 파일과 blob 단위로 동일함.
    - 커밋 메시지 안의 해시 참조 4개는 걸러진 커밋을 가리켜서 그대로 남음.
  - ⟦AI⟧ **GitHub 현황:** fork 0, issue 0, Actions·Release·deployment 0. `share.streamlit.io` push webhook 1개. `Shake2313/GABES`는 현재 저장소로 redirect됨(옛 이름). 새 `GABES`를 만들면 이 redirect는 사라지고, 문서에서 이 URL을 쓰는 곳은 0건.
  - ⟦AI⟧ **Streamlit 공식 문서:**
    - 앱은 GitHub owner·repo·branch·entrypoint로 식별됨. 미리 준비하지 않고 바꾸면 관리 권한을 잃음.
    - 저장소를 바꾸는 방법은 삭제 → GitHub 변경 → 재배포.
    - 앱을 지운 직후에는 subdomain을 다시 쓸 수 있음.
    - 현재 앱 주소는 `fwm-squeezing-app-bedekxcpwge62hjdkvelw8.streamlit.app`이고, User Guide 두 파일이 이 주소를 씀.
  - ⟦AI⟧ **S3 순서안:**
    1. 새 `GABES` 저장소 생성, filter-repo 결과 push.
    2. 새 저장소로 앱을 임시 주소에 배포하고 확인.
    3. 기존 앱 삭제 → 새 앱 subdomain을 기존 주소(또는 새 주소)로 변경.
    4. 기존 저장소의 webhook 제거 → 이름을 바꾸고 비공개로 전환.
    - 기존 앱이 남아 있는 상태에서 저장소 이름을 먼저 바꾸면 앱 관리 권한을 잃음.
  - ⟦AI⟧ **링크 매핑 초안:** 참조 2,134개 중 인용 1,969개는 유지, 링크 162개는 수정, `docs/papers`로 3개.
- 2026-11-09: **S3 결정** (사용자, 질문 응답).

  | ID | 결정 |
  |---|---|
  | R9 | 아카이브 저장소 이름 `GABES-archive` 확정. 아카이브 링크 형식: `https://github.com/Shake2313/GABES-archive/blob/pre-split-2026-11-09/<path>` |
  | R10 | **현재 폴더**(`G:\내 드라이브\Coding\GABES - Generic Atomic Bloch Equation Solver`)를 main 작업본으로 씀. 아카이브 작업본은 새 폴더로 옮김. 지금 경로에 묶인 Claude 메모리·Codex 설정·launch 설정은 main 쪽에 그대로 남음. |
  | R11 | Streamlit 앱 주소를 **새 주소로** 바꿈. 후보는 `gabes`(사용 가능할 때). 옛 자동 주소 링크는 끊기므로 User Guide·README를 새 주소로 고침. |

  - ⟦AI⟧ R10 로컬 전환 계획(S3-L). S3에서 push하기 전에 실행함.
    1. 아카이브 작업본: 현재 `.git`(전체 히스토리)에서 새 폴더(예: `G:\내 드라이브\Coding\GABES-archive`)로 `git clone --no-local`. remote는 `GABES-archive`.
    2. ignore·미추적 로컬 파일 이동(2026-11-09 조사, 총 8.6 GB, 18,134개):
       - 아카이브 쪽 6.84 GB → 아카이브 폴더로. 단 아래 검증용 사본은 별도로 결정.
       - main 쪽 172 MB는 남김: references 원문 PDF, `__pycache__`, UI 스크린샷.
       - 분류되지 않은 1.62 GB(`tmp/`, `.playwright-cli`, `.pytest_cache`)는 남기거나 지움.
       - **검증용 저장소 사본**(`output/ui_caption_commit_20261030` 2.19 GB, `output/ui_release_20261030` 2.18 GB, `output/commit_ui_20261022` 2.14 GB, `tmp/fwm-tpd-commit-20261002-55375359` 0.86 GB, `tmp/gain-hotfix-commit-check`, `tmp/checkpoint-validation-20260914` 등)은 지울지 옮길지 **사용자 결정 필요**.
         - **2026-11-11 사용자 결정: 삭제.** ⟦AI⟧ 저장소 스냅샷(`candidate/`, `.snapshot/`, `baseline/`, `parent/`)과 zip·tar, 전체 사본 2곳을 지움: 30개 항목, 7.94 GB, 파일 10,295개, 실패 0. 추적 파일 중 이 폴더들을 참조한 것은 이 허브와, 검증 위치를 적은 기록 두 곳(`output/rydberg_opacity_20261023/mirror_parity.json`, `analysis/pole_reuse/GATE.md`)뿐이었음.
         - ⟦AI⟧ 남긴 것: 커밋 준비 폴더 4곳의 작은 기록(메타데이터·patch·pytest 로그·스크립트, 약 5.4 MB)과 `tmp/gain-hotfix-commit-check`의 로그 2개. S3-L에서 아카이브 폴더로 옮김. Drive 휴지통에서 30일간 복구 가능.
    3. 현재 폴더의 `.git`을 filter-repo로 만든 main 저장소로 교체함. 이때 아카이브 쪽 추적 파일(이미 아카이브 작업본에 있음)을 지움.
       - 지우기 전에 아카이브 작업본의 HEAD가 같은 커밋이고 파일이 모두 있는지 확인함.
       - 교체는 되돌릴 수 있게 함: 옛 `.git`은 아카이브 폴더 clone이 끝날 때까지 보존.
    4. 새로 연결해야 하는 것: Codex weekly-review 자동화와 GC 작업 세션은 아카이브 폴더로. Claude 메모리는 main 경로에 남음.
- 2026-11-09: **S3 문서 준비** (사용자 "문서 정리 시작").
  - [`docs/papers/`](../papers/README.md): 세 문서 최신본을 바이트 그대로 복사(1.63 MB). README 출판 표는 여기로 연결함.
    - 아카이브 쪽 `test_publication_records`는 main 사본과 원본의 바이트 일치를 확인하도록 바꿈.
    - main 쪽 `tests/test_papers.py` 추가.
  - ⟦AI⟧ `tools/split_boundary.py` 확장:
    - `links [--apply]`: main 문서에서 아카이브를 가리키는 링크를 `GABES-archive/blob|tree/pre-split-2026-11-09/…`로 바꿈. 미리보기 기준 8개 파일 55개. 추적되지 않은 로컬 출력 링크 22개는 그대로 둠. 인용 목적 문서(이 허브, fixtures, papers)는 대상에서 뺌.
    - `history-paths`: filter-repo 경로 목록을 만듦. 리허설과 같은 647개.
    - `manifest`: Split Manifest. 리허설 commit-map으로 확인함(재작성 195, 제외 11).
  - ⟦AI⟧ AGENTS.md 초안 두 벌. 분리 전 GC 세션이 지금 AGENTS.md를 쓰므로 S3·S4에서 교체함.
    - [`AGENTS.main.md`](AGENTS.main.md): GC 절을 빼고 아카이브 안내, fixture 규칙, 크기 감시를 넣음.
    - [`AGENTS.archive.md`](AGENTS.archive.md): GC 절, engine submodule, R8, docs/papers 갱신 절차. Codex 검토 필요.
  - ⟦AI⟧ **S3 실행 순서 (runbook):**
    0. 선행 조건: Codex GC 작업 착지(473017f·0fab129 완료), 작업트리 깨끗, 검증용 사본 정리(2026-11-11 삭제 완료), `AGENTS.archive.md` Codex 검토.
    1. 마지막 문서 커밋:
       - `links --apply` 실행.
       - `AGENTS.main.md` → `AGENTS.md`.
       - README Deploy 절을 새 저장소 주소로 바꾸고 `test_docs_consistency`의 공개 문구도 함께 고침.
       - main 전용 checkout과 원래 저장소 양쪽에서 전체 테스트.
    2. 그 커밋에 태그 `split-source-<date>`.
    3. 아카이브 작업본: `git clone --no-local <현재 폴더> <G:\…\GABES-archive>`.
    4. 필터링:
       - 새 bare clone에서 main 외의 ref를 지움.
       - `history-paths` → `git filter-repo --paths-from-file` → `manifest`.
       - 필터링 결과로 main 전용 테스트.
    5. GitHub에 `Shake2313/GABES`(공개)를 만들고 push.
    6. Streamlit:
       - 새 앱을 임시 주소에 배포하고 확인.
       - 기존 앱 삭제 → 새 앱 subdomain 지정(R11, 후보 `gabes`).
    7. 기존 저장소:
       - webhook 제거 → 이름을 `GABES-archive`로 바꾸고 비공개로 전환.
       - 아카이브 작업본에서 로컬 커밋과 태그(`pre-split-2026-11-09`, `split-source-…`)를 push.
    8. 로컬 전환(R10, S3-L):
       - 현재 폴더의 `.git`을 필터링된 main으로 교체(옛 `.git`은 보관).
       - 아카이브 쪽 추적 파일 삭제, ignore 파일 이동.
    9. `coarse.jsonl`을 Release에 올리고 `external.json`에 기록.
    10. Codex 자동화와 GC 세션을 아카이브 폴더로 연결. README·User Guide의 앱 주소를 고쳐 main에 push.
- 2026-11-12: **S5 2단계 완료: `gabes_lab`과 pytest 전용 (R6, R7)** (사용자 "코덱스 작업이 끝났습니다. 이거 확인하고, 포함해서 2단계 진행해주세요").
  - ⟦AI⟧ Codex R13 확인 (c67d884, FWM affine replay 퇴역, 로컬):
    - 두 재현 플래그가 모두 true일 때만 재현 요청으로 봄. 공개 경로 세 곳(`compute_spectrum`, `full_spectrum`, `FWMScheme.compute`)과 provenance가 같은 함수 `reject_retired_replay`로 거절하고, 오류 문구가 아카이브 태그 `pre-split-2026-11-09`를 안내함.
    - 옛 설정 가져오기는 `Scheme.migrate_default_params`가 두 플래그를 지우고 나머지 값을 유지함. 둘 다 true면 `DefaultMigrationError`로 사유를 보여 줌.
    - 재현이 아닌 경로의 계산은 그대로임. 플래그 하나만 켠 호출도 provenance 필드가 이전과 같은 값임.
    - HEAD 2151a28에서 FWM·DEFAULTS·회귀·문서 테스트 604 통과.
    - Codex 인계의 아카이브 쪽 할 일: 엔진 고정 커밋을 c67d884 이후로 올릴 때 측정 이득 기록, Stark·Zeeman 소스 영수증, GC 감사를 다시 검증하고, 옛 재현 스위트는 `pre-split-2026-11-09`에서 돌림.
  - ⟦AI⟧ 커밋(로컬, push 없음):
    - 6e3ee82: `gabes_lab/` 생성과 이동(R6).
      - `blocks/paths.py`의 `REPO_ROOT`는 위로 올라가며 `gabes/`와 `pytest.ini`가 함께 있는 폴더를 찾음. 아카이브의 `engine/` 안에서도 같은 방식으로 찾음.
      - `pipelines/rydberg_cell_heating`, `pipelines/squeezing`(하위 `low_pump_power`, `variable_audit`), `tools/`(`ui_audit`, PSR 벤치마크 2개, `split_boundary`, `migration_manifest`).
      - rydberg 패키지 안의 테스트 2개는 `tests/test_rydberg_cell_heating_workflow.py`, `tests/test_rydberg_cell_heating_adapters.py`로 옮김.
      - 실행은 `python -m gabes_lab.<area>.<module>`. 이동한 스크립트의 `sys.path.insert`와 직접 실행용 import 우회를 지움.
      - 산출 경로는 바꾸지 않음(결과 유지 원칙). squeezing 스캐너는 계속 `analysis/squeezing/...`에 쓰므로 `.gitignore`에 `/analysis/`와 `gabes_lab/**/generated*/`를 넣고, 아카이브로 간 npz 예외 줄을 지움. 산출 위치 정리는 3단계 io·runner 블록에서 함.
      - README 폴더 지도에 `gabes_ui/`, `sabes/`, `gabes_lab/`를 넣고, AGENTS·UI 허브·`gabes_ui/DESIGN.md`의 도구 경로를 고침.
    - 5b4a96d: pytest 전용(R7). `pytest.ini`에 `pythonpath = .`를 넣고 테스트 35개에서 `sys.path.insert`를 지움(쓰지 않게 된 `ROOT`·`sys`·`Path` import도 함께). README 테스트 안내는 모두 `python -m pytest -q`. 직접 실행하는 스크립트 `tests/capture_baseline.py`, `tests/verify_hanle_eit_eia.py`는 그대로 둠.
    - 이 기록 커밋: Migration Manifest 재생성(18b0b5f 이후 52건: 그대로 이동 33, 이동+경로 수정 19).
  - ⟦AI⟧ 설계 세부 조정: 개발 도구는 `gabes_lab/tools/`에 둠(R6의 "tools 흡수"). 테스트 파일의 `if __name__ == "__main__"` 블록은 남김. 이제 직접 실행되지 않으므로 정리 후보임.
  - ⟦AI⟧ 검증: 이동 관련 테스트 62 통과, main 전체 2,367 통과 / 1 skip / 0 실패(330 s). `python -m`으로 `ui_audit`, rydberg `workflow`, `migration_manifest` 실행 확인. `--help`가 CP949 콘솔에서 실패하는 문제는 이동 전부터 있었음(UTF-8 출력에서는 정상).
  - 다음: 3단계 블록(provenance, io, figures, snapshot, runner, publish_to_archive), 4단계 AGENTS 규칙과 `gabes_lab/README.md`.
- 2026-11-12: **S5 시작, 1단계 UI 파일럿 완료** (사용자 "S5 시작").
  - ⟦AI⟧ reflog: Drive가 `.git/logs` 생성을 다시 허용함. `core.logAllRefUpdates` 설정을 지우고(기본값 true) 임시 ref로 기록되는 것을 확인함.
  - ⟦AI⟧ 옛 `.git` 백업 `GABES-pre-split-dotgit`(3.0 GB)을 사용자 승인으로 지움. 지우기 전에 확인한 것:
    - 브랜치·태그 11개는 모두 아카이브 원격에 있음.
    - 아카이브에 없던 것은 Codex 앱의 turn-diff 스냅샷 ref 37개와, 지난 커밋 준비 때 에이전트가 `.git` 안에 쓴 작업 파일(임시 index·상태 파일·스크립트)뿐임.
    - `grand-challenge-runs`는 thermal campaign 실행기의 `TemporaryDirectory`가 남긴 소스 사본과 pyc뿐이었음.
  - ⟦AI⟧ Codex가 같은 작업본에서 R13을 진행 중이어서, S5는 Codex가 고치는 파일(`README.md`, `docs/checklist.json`, FWM 엔진·스킴, `gabes_ui/defaults*.py`, 관련 테스트)을 건드리지 않음. 커밋은 `git commit --only -- <내 경로>`로 내 파일만 담음.
  - ⟦AI⟧ 커밋(로컬, push 없음):
    - 53c35db: 자산 경로를 `gabes_ui/paths.py` 한 곳으로 모음.
    - 3996056: R5 이동. `assets/` → `gabes_ui/assets/`, `sabes/components/`·`sabes/layout/render.py`·`sabes_page.py` → `gabes_ui/sabes/`. 루트에는 `streamlit_app.py`, `.streamlit/`, `static/`만 남음.
    - 이 기록 커밋: [디자인 허브](../../gabes_ui/DESIGN.md), 스킬 `.claude/skills/gabes-ui-design`, Migration Manifest 생성기 [`tools/migration_manifest.py`](../../tools/migration_manifest.py)와 [첫 결과](s5_migration/migration_manifest.json)(25건: 그대로 이동 22, 이동+경로 수정 3).
  - ⟦AI⟧ 설계 세부 조정: 계획은 "이동과 내용 수정은 별도 커밋"이었으나, 이동 커밋에 import·경로 수정을 함께 넣음. 중간 커밋이 깨지지 않게 하려는 것이며, 수정이 작아 git이 rename으로 인식함(유사도 84–100 %).
  - ⟦AI⟧ 검증: 관련 SABES·UI·문서 테스트 473 통과. main 전체 2,367 통과 / 1 skip / 0 실패(345 s, Codex R13 진행 중 변경이 작업트리에 있는 상태). 내장 브라우저에서 첫 화면과 SABES 광학 테이블 확인.
  - 다음: 2단계(`gabes_lab` 생성, `tools/`와 재사용 파이프라인 이동, 루트 conftest, 테스트 37개의 `sys.path.insert` 제거, README 실행 안내)는 R13 착지 뒤. 이어서 3단계 블록, 4단계 AGENTS 규칙.
- 2026-11-11: **S4 완료 (Codex d1a74d8), 독립 검증, 아카이브 push** (사용자 "S4 진행", push 승인 "지금 push").
  - Codex d1a74d8 (아카이브):
    - main 경로 626개를 제거함.
    - `engine/` submodule을 GABES f333a50에 고정함.
    - 아카이브 루트에 `AGENTS.md`를 설치함.
    - 근거 기록은 아카이브 `docs/grand_challenge/archive_s4_2026_11_11.md`.
    - weekly-review 자동화를 아카이브 작업본으로 옮김(05:00 Asia/Seoul). 사용자가 노트북의 자동화를 원격으로 옮김.
  - ⟦AI⟧ 고정된 엔진 f333a50과 Split Manifest `main_head` f89c7a2는 코드가 같음. 차이는 문서, User Guide, 분리 기록, `test_docs_consistency`뿐임.
  - ⟦AI⟧ 다른 경로에 clone한 사본(`core.longpaths=true`)에서 S4 확인 4개가 모두 통과함.
    - `gabes.core`가 `engine/gabes/core.py`에서 읽힘.
    - ledger reader의 봉인 입력 3개 해시가 일치함: plan 749f6f66…, sources.zip db9e039c…, grid-p3-s11 5f6075bf….
    - `tests/quantum/test_variance_reduction_pilot_shadow.py` 7개 통과.
    - GC 등록부는 아카이브 루트에 있음(`grand_challenges` 1, 항목 1). 제품 등록부는 `engine/docs/checklist.json`(항목 115, `grand_challenges` 없음). 분리 허브는 `engine/docs/repo_restructure/`.
  - ⟦AI⟧ 같은 사본에서 아카이브 HEAD 전체 테스트: **1,317 통과 / 64 실패 / 3 skip, 495 s**. 실행 후 상태는 깨끗함.
    - 데이터·봉인 기록 손상은 없음. 실패는 모두 엔진이 `gabes/`에서 `engine/gabes/`로 옮겨진 데서 생김.
    - 봉인 thermal campaign의 고정 소스 목록 검사 41개: ensemble 24, extend 14, grid·diagnose·cross_compare 각 1. `engine/gabes/__init__.py`를 목록 밖 소스로 거부함.
    - 엔진 소스를 `ROOT/gabes/…`에서 직접 읽는 테스트 23개: spatial_uncertainty_audit 11, FWM zeeman 검증 3, stark 검증 1, `tests/quantum` 감사 8개 파일 각 1. 대부분 FileNotFoundError로 실패함.
    - R8에 따르면 옛 증거 재현은 `pre-split-2026-11-09`에서 함. 아카이브 HEAD 테스트는 `engine/` 경로로 고치거나, 옛 배치 전용으로 표시하고 사유와 함께 건너뛰어야 함. Codex 담당이며, 작업본에서 수정이 진행 중임(미커밋).
  - ⟦AI⟧ 아카이브 작업본에서 `engine/` index가 비어 있었음. Drive가 남긴 `.git/modules/engine/index.lock`(19:47)을 git 프로세스가 없는 것을 확인한 뒤 지우고, HEAD에서 index를 다시 만듦. 파일은 바꾸지 않았고 상태는 깨끗함.
  - ⟦AI⟧ 사용자 승인 후 `GABES-archive` main에 d1a74d8을 push함(3da3b58 → d1a74d8). push 시점에 있던 Codex의 미커밋 수정은 보내지 않음.
  - Codex f9d1130 (아카이브, `test(archive): verify split engine layout`): 위 64개 실패를 수정함.
    - 바뀐 것은 테스트, fixture 1개, GC 기록 문서뿐임. 봉인 실행기·증거 파일, 엔진 고정 커밋 f333a50, 태그 `pre-split-2026-11-09`는 그대로임.
    - 엔진 소스를 확인하는 감사와 FWM 영수증 검사는 실제 `engine/` 소스를 읽음. 영수증의 소스 이름은 그대로 `gabes/…`임.
    - 합성 thermal campaign 테스트는 입력 소스 목록에 `engine/gabes` 경로를 넣어 기존 해시 검사와 실패 조건 검사를 그대로 실행함.
    - 원래 소스 배치가 필요한 새 하위 프로세스 검사 8개에는 `pre_split_layout`을 붙여, 재현 위치(`pre-split-2026-11-09`)를 사유로 적고 건너뜀. 그중 compare 검사 1개는 d1a74d8에서 `gabes` import 오류를 의도한 거부로 잘못 받아들여 통과하던 검사임.
    - Zeeman Kim 고정점 검사용으로 Stage-1 캐시 1개(6,058 B)를 `tests/fixtures/fwm_zeeman_stage1/`에 복사함. 이 캐시는 git에서 추적하지 않던 파일이라 새 clone에서 실패했음.
  - ⟦AI⟧ f9d1130 독립 재검증 (같은 다른 경로 사본, Python 3.14.2, `python -B -m pytest -q -rs --durations=20 -p no:cacheprovider`): **1,373 통과 / 11 skip / 0 실패, 491 s**. 실행 후 상태는 깨끗함.
    - skip 11개 = `pre_split_layout` 8 + Windows 심볼릭 링크 권한이 필요한 기존 검사 3.
    - ledger reader의 봉인 입력 3개 해시가 다시 일치함.
    - fixture는 아카이브 로컬 캐시 원본과 sha256이 같음(7ef597f7…). 테스트가 이 해시와 캐시 키를 확인함.
    - 테스트 수정으로 약해진 assertion은 없음. 엔진 소스 해시를 기록값과 대조하는 검사가 새로 붙음.
  - ⟦AI⟧ 사용자 승인("둘 다 push") 후 `GABES-archive` main에 f9d1130을 push함(d1a74d8 → f9d1130). 검증용 사본(2.9 GB)은 미push 내용이 없는 것을 확인한 뒤 지움.
  - 다음: S5(main 통합).
- 2026-11-11: **Codex S4 인계 반영** (사용자가 Codex 답장 전달).
  - 473017f(ledger reader)·0fab129(GC 아카이브 재현 규칙) 확인.
    - 원래 경로에서 `python -m analysis.grand_challenge.variance_reduction_ledger_reader` 통과(입력 3개 해시 일치).
    - GC 관련 테스트 54개 통과. 다른 경로 검증은 Codex가 보고한 결과이고, S4에서 다시 확인함.
  - **출처 정정 (⟦AI⟧):** checklist의 `fwm-legacy-gain-replay-retirement.recommendation_2026_11_09`와 `gc-absolute-path-ledger-reader` done 표시는 Codex가 쓴 내용임.
    - 다만 Claude 커밋 7977b22(18:05:20)에 같이 들어감. Claude가 작업 파일의 checklist를 수정할 때 Codex의 미커밋 수정이 이미 있었음.
    - 내용은 바뀌지 않았음.
  - 사용자 결정 (2026-11-11):

    | ID | 결정 |
    |---|---|
    | R12 | **GC 등록부는 아카이브의 별도 파일.** `docs/checklist.json`의 `grand_challenges` 절(127 KB)과 GC 항목을 `docs/grand_challenge/checklist.json`으로 옮김. main checklist에는 제품 항목만 남음(옛 내용은 히스토리에 남음). GC 규칙·메모리 링크를 새 파일로 연결. |
    | R13 | **FWM legacy affine replay는 분리 후 main에서 퇴역**(Codex 권고안). 옛 설정값은 받아들이되, replay를 명시적으로 요청하면 퇴역 사실과 아카이브 태그를 알려 줌. 재현은 아카이브 `pre-split-2026-11-09`에서 유지. |

  - ⟦AI⟧ **R12 실행 (2026-11-11, 사용자 "지금 분리"):**
    - `grand_challenges` 절과 GC 항목 `gc-absolute-path-ledger-reader`를 [`docs/grand_challenge/checklist.json`](../grand_challenge/checklist.json)으로 옮김.
    - main checklist는 해당 블록만 삭제함(635 KB → 472 KB, 항목 116 → 115). 다른 키와 항목은 그대로임.
    - 옮긴 값은 원본과 동일함. `grand_challenges`를 읽는 실행 코드는 봉인 스크립트 `mean_gain_2026_10_15/close_session.py`뿐이며, pre-split 태그에서 재현하므로 고치지 않음.
    - 현재 AGENTS.md와 두 초안에 새 위치를 적음. GC 규칙·head-manager 스킬·memory 링크는 Codex 담당.
  - ⟦AI⟧ **S4 확인 항목 (Codex 요청):**
    1. `engine/`이 Split Manifest의 main 커밋을 가리키는지, GC 테스트가 그 엔진을 import하는지, 소스를 해시하는 검증기가 실제 `engine/` 경로를 읽는지 확인.
    2. 다른 경로에 checkout한 아카이브 루트에서 ledger reader와 `tests/quantum/test_variance_reduction_pilot_shadow.py` 실행(봉인 입력 검증). 역사적 재현은 `pre-split-2026-11-09`(0c95124)의 당시 경로·실행기로.
    3. GC 등록부 위치(R12)에 맞춰 아카이브 규칙과 메모리 링크 연결.
    4. weekly-review 자동화 이전은 아카이브와 고정 `engine/` 검증 뒤에. 05:00 Asia/Seoul, 요일 배정, 모델 설정 유지.
- 2026-11-11: **R10 로컬 전환 완료 → S3 종료** (사용자 "로컬 전환 진행").
  - ⟦AI⟧ 원래 폴더(`G:\내 드라이브\Coding\GABES - Generic Atomic Bloch Equation Solver`)는 이제 **GABES main 작업본**(origin `Shake2313/GABES`, HEAD f333a50, 추적 파일 635개).
  - ⟦AI⟧ ignore·미추적 아카이브 쪽 로컬 파일 788개 항목(파일 5,321개, 514.7 MB)을 `G:\내 드라이브\Coding\GABES-archive`의 같은 경로로 이동. main 쪽 142개와 `.pytest_cache`는 유지.
  - ⟦AI⟧ 옛 `.git`(c1b0777)은 `G:\내 드라이브\Coding\GABES-pre-split-dotgit`에 보관.
  - ⟦AI⟧ main `.git`은 폴더 안에서 `git init` + `GABES` fetch로 만듦. Drive가 옛 `.git/logs` 삭제를 끝내지 않아 `core.logAllRefUpdates=false`를 임시로 설정함(정리 후 다시 켬).
  - ⟦AI⟧ 아카이브 쪽 추적 파일 4,783개(2.48 GB)는 아카이브 작업본과 sha256 일치(4,531개) 또는 c1b0777 git blob 일치(252개, 줄바꿈 형식만 다름)를 확인한 뒤 삭제. 남긴 파일 0개. 중첩 저장소 `fwm_quotient_site/.git`은 아카이브로 이동.
  - ⟦AI⟧ main 폴더 전체 테스트: 2,355 통과 / 1 skip / 0 실패(360 s). 무시 처리된 로컬 참고 프로그램 `references/AutoOD-Rb/`의 테스트 78개 포함. 실행 후 상태 깨끗함.
  - 다음: 아카이브 쪽 S4(main 경로 제거, `engine/` submodule을 GABES f333a50 이후로 고정, `AGENTS.archive.md` 설치, Codex의 S4 확인 4개). weekly-review 자동화는 아카이브 작업본으로 이전.
- 2026-11-11: **S3 공개 단계 완료** (사용자 승인 "승인 받는 작업 진행합시다", Streamlit은 사용자가 패널에서 로그인).
  - `Shake2313/GABES`(공개)를 만들고 main을 push. 공개 전 점검에서 새로 드러나는 개인 정보는 없었음.
  - Streamlit: 새 앱 `gabes-app.streamlit.app`을 배포(`gabes`는 6자 미만이라 사용자가 `gabes-app` 선택, Python 3.14). SAS 스펙트럼과 FWM 이득·잡음 그래프가 그려지는 것을 확인한 뒤 옛 앱(`fwm-squeezing-app-bedekxcpwge62hjdkvelw8`) 삭제. 옛 저장소 webhook은 앱과 함께 제거됨.
  - 옛 저장소를 `Shake2313/GABES-archive`로 이름 변경하고 비공개로 전환.
  - User Guide 두 파일과 README의 앱 주소를 새 주소로 갱신.
- 2026-11-11: **S3 로컬 단계 완료** (사용자 "로컬 단계 시작").
  - ⟦AI⟧ source `c1b0777`에 태그 `split-source-2026-11-11`. source 폴더는 여기서 동결.
  - ⟦AI⟧ 아카이브 작업본: `G:\내 드라이브\Coding\GABES-archive`(HEAD c1b0777, 태그 2개, 파일 5,415개). 내용은 `git diff`로 동일함을 확인. Drive에서 stat 불일치로 1,884개가 "수정됨"으로 보이는 표시는 별도 확인.
  - ⟦AI⟧ filter-repo: 경로 655개(규칙 635 + rename 계보 20) → 커밋 203개(제외 14), 압축 5.72 MiB. HEAD 트리 = source main 쪽 632개 파일과 blob 단위로 동일.
  - ⟦AI⟧ [Split Manifest](split_2026_11_11/split_manifest.json), 경로 목록, 공개 전 main 전체 테스트(2,277 통과 / 1 skip / 0 실패)를 `split_2026_11_11/`에 보관.
  - ⟦AI⟧ main 전용 커밋: `AGENTS.md` ← `AGENTS.main.md`, 아카이브 링크 55개를 `GABES-archive` 고정 URL로 치환, README의 GC 블록을 아카이브 안내로 교체하고 Deploy 절 갱신, User Guide의 저장소 주소와 clone 명령을 `GABES`로 갱신.
  - 다음 (사용자 승인 2026-11-11 "승인 받는 작업 진행합시다"): `GABES` 공개 저장소 생성·push → Streamlit 새 앱 → 기존 앱 삭제 → 기존 저장소 이름 변경·비공개 전환 → 아카이브 push·Release → 로컬 전환.
- 2026-11-11: **S3 선행 조건 충족.**
  - Codex de03478: `AGENTS.archive.md` 초안 검토·수정, GC 규칙·head-manager 스킬·hypermemory·memory.json을 R12 등록부로 연결.
  - Codex가 대조한 결과: `grand_challenges` 절 일치, 항목 116 = 제품 115 + GC 1(누락·중복 없음), `docs/papers` 사본 3개가 원본과 바이트 일치.
  - 선행 조건 상태: GC 작업 착지(473017f·0fab129·de03478), 검증용 사본 삭제, 초안 검토 모두 완료. 남은 미커밋은 weekly-review가 쓴 daily report뿐이며 S3 1단계에서 함께 커밋.
  - ⟦AI⟧ **S4 확인 항목 (Codex 최신판으로 대체):**
    1. 검토한 초안을 아카이브 루트 `AGENTS.md`로 설치. GC 등록부는 아카이브 루트에서, 제품 등록부(`engine/docs/checklist.json`)와 분리 허브(`engine/docs/repo_restructure/`)는 `engine/`에서 읽히는지 확인.
    2. `engine/`이 Split Manifest의 main 커밋을 가리키는지, GC 테스트의 `gabes` import와 엔진 소스 해시 검증기가 모두 그 checkout을 쓰는지 확인.
    3. 다른 경로에 checkout한 아카이브 루트에서 `python -m analysis.grand_challenge.variance_reduction_ledger_reader`와 `python -m pytest -q tests/quantum/test_variance_reduction_pilot_shadow.py`를 실행하고 결과를 기록(봉인 입력 검증). 역사적 재현은 `pre-split-2026-11-09`(0c95124)의 당시 경로·실행기로.
    4. weekly-review 자동화 이전은 위 검증 뒤에 (05:00 Asia/Seoul, 요일 배정, 모델 설정 유지).
  - R13(FWM affine replay 퇴역)은 분리 후 공개 main 작업.
- 2026-11-09: S2 진행 기록 (사용자 "S2 시작").
  - ⟦AI⟧ 경계 도구 [`tools/split_boundary.py`](../../tools/split_boundary.py)를 만듦.
    - 규칙은 순서대로 적용하고, 처음 맞는 규칙이 이김. 정적 참조 검사와 main 전용 sparse checkout도 함께 함.
    - HEAD 기준 main 608개 파일(12.2 MB), 아카이브 4,753개(2.48 GB).
  - ⟦AI⟧ main 전용 checkout 첫 실행: 2,154 통과, 46 실패, 21 오류. 원인 테스트 파일은 9개.
  - 사용자 결정 2026-11-09:
    - magneto 동결 기준 비교 96개는 기준 코드를 fixture로 복사해 main에 남김.
    - 퇴역한 FWM gain hotfix 테스트 50개는 아카이브로 보냄. legacy replay 코드를 정리할지는 checklist `fwm-legacy-gain-replay-retirement`로 따로 결정.
  - ⟦AI⟧ 처리:
    - [`tests/fixtures/`](../../tests/fixtures/README.md): atomic_qrt, gain_profile, magneto 기준 코드 6개. 바이트 복사이고, optical 검증 모듈만 import 한 줄을 고침.
    - 아카이브로 보내는 테스트: FWM 보정 검증 6개, gain hotfix 2개, `test_fwm_docs_consistency`, 새로 떼어낸 `test_publication_records`·`test_fwm_measured_gain_records`.
    - `tests/test_repo_size_guard.py`: main 쪽 파일당 1.5 MB(static 5 MB), 전체 50 MB.
    - `.claude/skills`와 `launch.json`을 추적함.
    - checklist `gc-absolute-path-ledger-reader`(S4 전에 해결) 등록.
  - ⟦AI⟧ 출처 경로 규칙: main의 코드·`gabes/data`가 인용하는 `analysis/…`, `docs/grand_challenge/…` 경로는 바꾸지 않음.
    - 바이트와 해시를 유지하기 위해서임.
    - 이 경로들은 아카이브 저장소의 `pre-split-2026-11-09`(또는 그 이후 커밋)에서 해석함. README 안내는 S3에서 함.
  - ⟦AI⟧ S3로 미룬 것: main 문서(README 80개, ui_redesign log 31개 등)가 아카이브 경로를 가리키는 링크 정리, R4의 `docs/papers/`, 두 저장소용 AGENTS.md 분리. 링크 대상이 아카이브 저장소의 최종 URL에 따라 정해지기 때문임.
- 2026-11-09: S1 진행 기록.
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
| S1 | 동결: 다른 세션 작업 착지. untracked analysis 1,499개 분류(아카이브 커밋/삭제). `(1)` 사본 정리. 태그 `pre-split-<date>`. 기준선 저장: full pytest, `tests/quantum`, GC 검증기. | 완료 2026-11-09 (`pre-split-2026-11-09`, 3,733 통과 / 0 실패) |
| S2 | 위 의존 끊기, .gitignore·크기 감시, `output/` untrack → full pytest | 완료 2026-11-09 |
| S3 | GitHub 이름 변경 → 아카이브 비공개. `git filter-repo`로 main 경로 추출 → 새 `GABES` push. Split Manifest(경로 목록, commit-map) 생성. Streamlit Cloud 재연결. README·User Guide URL 갱신. | 완료 2026-11-11 |
| S4 | 아카이브: main 경로 제거, submodule `engine/`, R8 적용. 분리 태그에서 `tests/quantum`·GC 검증기 재실행. | 완료 2026-11-11 (d1a74d8, f9d1130: 아카이브 HEAD 1,373 통과 / 0 실패, 옛 배치 검사 8개는 분리 태그에서 재현) |
| S5 | main 통합 (이전 P2–P5) | 진행 중: 1단계 UI 파일럿, 2단계 `gabes_lab`·pytest 전용 완료 2026-11-12 |
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
