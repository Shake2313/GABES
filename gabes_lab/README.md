# gabes_lab 개발 패키지

분석 스크립트와 개발 도구가 함께 쓰는 코드를 모은 곳. 물리는 `gabes`에 있고, 여기에는 그 주변에서 반복되는 일을 둠: 경로, 출처 기록, 파일 쓰기, 그림, 실행 폴더, 리팩터 전후 비교, 아카이브 반출.
이 구조의 결정 이력은 [저장소 분리 허브](../docs/repo_restructure/README.md)(R6·R7, S5 2–4단계)에 있음.

모듈은 저장소 루트에서 `python -m gabes_lab.<area>.<module>`로 실행함. 테스트는 pytest로만 실행함(`python -m pytest -q`; `pytest.ini`의 `pythonpath = .`가 import 경로를 맞춤).

## 패키지 지도

| 위치 | 내용 |
|---|---|
| `blocks/` | 물리 없는 작은 도우미. 아래 블록 표. |
| `pipelines/rydberg_cell_heating/` | Rydberg 셀 가열 검증 흐름: 설정 기반 온도 스윕, 측정 CSV 겹쳐 그리기, 출처 매니페스트, 그림, XeLaTeX 보고서 입력. 사용법은 그 폴더의 [README](pipelines/rydberg_cell_heating/README.md). |
| `pipelines/squeezing/` | 85Rb D1 이중 Λ FWM squeezing 스캐너. 결과는 `runs/squeezing/<scan>/`. |
| `tools/` | 개발 도구: `ui_audit`(화면 측정), `benchmark_psr_thermal`·`benchmark_psr_propagation`, 저장소 관리 `split_boundary`·`migration_manifest`. |

## 블록

| 블록 | 쓰는 때 | 주요 이름 |
|---|---|---|
| `paths` | 저장소 루트가 필요할 때. `Path(__file__).parents[n]` 대신 씀. | `REPO_ROOT`, `find_repo_root` |
| `provenance` | 파일 해시, git 상태, 저장소 밖 경로 숨기기, 시각, 런타임 버전 | `sha256_file`, `sha256_bytes`, `file_record`, `portable_path`, `git`, `git_state`, `github_slug`, `utc_now`, `runtime_info` |
| `io` | UTF-8 텍스트·JSON 쓰기와 읽기. 부모 폴더를 만들고 `ensure_ascii=False`로 씀. | `write_text`, `write_json`, `dumps_json`, `read_json` |
| `figures` | 화면 없는 matplotlib, 형식별 그림 저장 | `pyplot`, `save_figure` |
| `runner` | 실행마다 새 출력 폴더와 `run.json` 기록 | `start_run`, `Run` |
| `snapshot` | 결과를 지켜야 하는 리팩터의 전후 비교 | `capture`, `save`, `load`, `compare` |
| `publish_to_archive` | 끝난 결과를 GABES-archive 작업본으로 복사하고 매니페스트 작성 | `publish_to_archive`, `check_archive` |

### 자주 쓰는 형태

새 스캔 스크립트:

```python
import numpy as np

from gabes_lab.blocks import figures
from gabes_lab.blocks.io import write_json
from gabes_lab.blocks.runner import start_run


def main(argv=None):
    args = parse_args(argv)                         # argparse, --out 포함
    with start_run("squeezing/my_scan", out=args.out, config=vars(args)) as run:
        result = compute(args)                      # 물리는 gabes에서 import
        np.savez(run.path("my_scan.npz"), **result)
        write_json(run.path("summary.json"), summarize(result))
        plt = figures.pyplot()
        fig, ax = plt.subplots()
        ...
        figures.save_figure(fig, run.path("my_scan"), ("png", "pdf"), dpi=150)
```

- 기본 출력은 `runs/<이름>/<UTC 시각>-<커밋 7자리>/`이고 git에 들어가지 않음. `--out`을 주면 그 폴더에 씀.
- `run.json`(`gabes_lab.run/1`)에는 인자, 설정, 출력 전에 잡은 git 상태, 런타임 버전, 소요 시간, 출력 파일 sha256이 남음. `with` 안에서 예외가 나면 status가 `failed`가 되고, 프로세스가 죽으면 `running`으로 남음.

리팩터 전후 비교([AGENTS 통합 규칙](../AGENTS.md#unified-engine-development-principle): 비트 동일, 또는 상대 1e-12 이하이면서 표시값 동일):

```python
from gabes_lab.blocks import snapshot
snapshot.save("before.npz", snapshot.capture(outputs))     # 고치기 전
snapshot.save("after.npz", snapshot.capture(outputs))      # 고친 뒤
```

```bash
python -m gabes_lab.blocks.snapshot compare before.npz after.npz --rtol 1e-12
```

`capture`는 dict, list, dataclass, 객체의 공개 속성을 키 경로(`a/b/0`)로 펼침. 종료 코드 0이면 허용 범위 안.

결과를 아카이브로 보낼 때:

```bash
python -m gabes_lab.blocks.publish_to_archive runs/squeezing/my_scan/<run> --archive ../GABES-archive --dest analysis/squeezing/2026-11-12_my_scan --dry-run
```

- 미완료 실행, 실행 뒤 바뀐 파일, 커밋 안 된 main(`--allow-dirty`를 주면 dirty로 기록), 비어 있지 않은 대상, `engine/` 안 경로는 거절함.
- 매니페스트(`gabes_lab.publish/1`)에 main 저장소·커밋, 아카이브 engine 고정 커밋과 일치 여부, 파일 sha256이 남음.
- 아카이브에 커밋하지 않음. 커밋은 아카이브 쪽에서 검토 후 함.

## 규칙

- 블록을 먼저 찾고 import함. 맞는 블록이 없으면 스크립트 안에 새로 씀. 철칙은 아님: 필요한 분석이 지금 구조에 없으면 새로 만들어도 됨.
- 같은 도우미가 두 번째 호출부에서 필요해지고 물리가 없으면 블록으로 옮김. 기존 호출부도 같이 옮기고, 결과는 `snapshot`이나 바이트 비교로 같음을 확인함.
- 블록에 물리를 넣지 않음. 원자 데이터, 해석식, 수치 풀이는 `gabes`의 공유 모듈에 둠([AGENTS 통합 규칙](../AGENTS.md#unified-engine-development-principle)).
- `gabes`와 `gabes_ui`는 `gabes_lab`을 import하지 않음. 의존은 `gabes_lab` → `gabes` 한 방향.
- 새 결과는 `runs/`에 씀. main의 `analysis/`나 `docs/` 결과 폴더에 쓰지 않음. 결과가 main 밖으로 나가는 길은 `publish_to_archive` 하나임. `tests/test_repo_size_guard.py`가 main 크기를 감시함.
- 기존 기록 파일을 다시 쓸 때는 바이트를 지킴. `write_json`의 `indent`·`allow_nan`·`default`·`trailing_newline` 인자가 그 용도임. 텍스트 모드라 Windows에서는 줄바꿈이 CRLF임.
- `sys.path.insert`를 쓰지 않음. 직접 실행하는 스크립트는 `python -m`으로 실행함. 예외: 스킬 폴더처럼 패키지 밖에서 직접 실행하는 스크립트.
- 스크립트 docstring에 실행 명령, 입력, 출력 위치를 적음.

## 새 블록·파이프라인·도구

- 블록: `blocks/<이름>.py`. 표준 라이브러리, numpy, 그림이면 matplotlib만 씀. 테스트는 `tests/test_lab_blocks.py`에 추가하고, 이 README의 블록 표와 `blocks/__init__.py` docstring에 한 줄씩 넣음.
- 파이프라인: `pipelines/<영역>/`. 출력은 `runner`로 `runs/<영역>/<이름>/`에 쓰고 `--out`을 받음. 물리는 `gabes`의 공유 모듈을 씀.
- 도구: `tools/<이름>.py`, `main(argv=None)`으로 테스트에서 부를 수 있게 함.
- 기존 Rydberg 흐름은 자체 출력 폴더(`generated*/`, git 제외)와 `source_manifest.json`을 유지함. 출처 필드는 `provenance` 블록으로 만듦.

## 검증

1. 블록을 고치면 `python -m pytest -q tests/test_lab_blocks.py`와 그 블록을 쓰는 호출부의 테스트.
2. 호출부를 블록으로 옮기면 같은 입력으로 전후 출력을 비교함(`snapshot compare`, JSON·PNG 바이트). 실행 시각처럼 매번 바뀌는 값만 차이로 허용함.
3. 출력 위치를 바꾸면 기본 위치로 한 번 실행해 `run.json`이 `complete`인지 확인함.
