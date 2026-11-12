# gabes_ui 디자인 허브

UI 패키지에서 무엇이 어디 있는지와, 코드를 고칠 때 지킬 규칙을 모은 곳.
결정 이력·단계 계획·검수 기록은 [UI 리디자인 허브](../docs/ui_redesign/README.md)에 있음.
SABES 광학 테이블의 추가 규칙은 [SABES 디자인 규칙](../docs/SABES/design_rules.md).

## 패키지 지도

| 위치 | 내용 |
|---|---|
| `theme.py` | 색 토큰 `LIGHT`·`DARK`, 글꼴 IBM Plex Sans/Mono, CSS 변수 `--g-<name>`. 색과 글꼴의 단일 출처([D2](../docs/ui_redesign/decisions.md)). `.streamlit/config.toml`은 Streamlit 위젯에 필요한 몇 값만 반복함. |
| `paths.py` | 자산 위치 `ASSETS`·`UI_ASSETS`와 읽기 함수 `asset_text`. 자산 경로는 여기서만 계산함. |
| `assets/` | 로고·마크 SVG. `assets/ui/`에 공용 스타일시트 `gabes.css`와 컴포넌트 JS·CSS. |
| `format.py`, `units.py` | 숫자 표시 규칙(D7), 단위가 붙은 입력 파서. |
| `scrub.py` | ScrubField 정밀 입력(D3, D9). |
| `controls.py`, `conditions.py` | rail 조립, 메인 화면 Conditions 팝오버(D17). |
| `shell.py`, `plotcard.py`, `readout.py`, `export.py`, `guide.py` | 상단 바, 그림·오버레이·More, readout strip, 내보내기, 안내서 버튼. |
| `liveplot.py`, `model_compare_plot.py`, `model_compare_section.py`, `spectral_compare.py`, `spectral_sweep.py`, `plot_focus.py` | 스펙트럼·지도 탐색 컴포넌트. |
| `component_runtime.py`, `dropdowns.py`, `update_status.py`, `diagnostics.py` | Streamlit v2 컴포넌트 공용 런타임, 드롭다운 정책, 갱신 상태 표시, 진단. |
| `defaults.py`, `defaults_browser.py`, `defaults_store.py` | 이름 붙인 DEFAULTS(D14). |
| `sabes/` | SABES 화면. `page.py`는 `?app=sabes` 진입점 `render(host=...)`, `render.py`는 광학 테이블 그리기 사양, `components/`는 SVG 캔버스 컴포넌트. SABES 모델(광로·계측기·배치 기하)은 루트 `sabes/` 패키지. |

## 규칙

- UI는 물리를 계산하지 않음. scheme이 선언한 `param_schema`를 그리고 `observables`를 표시함. 계산 경로를 고치는 UI 변경은 없음.
- 색·글꼴은 `theme` 토큰만 씀. 컴포넌트 iframe에도 토큰을 넘김. 새 하드코드 색을 만들지 않음.
- 자산은 `gabes_ui.paths`로 찾음. 모듈 위치에서 `Path(__file__).parents[...]`로 자산을 찾지 않음.
- 새 화면 요소는 기존 요소를 하나 이상 대체하거나 조건부로만 보임(감량 원칙 D4).
- UI 라벨·단위는 unicode(°C, µW, Γ, 2π). matplotlib 축 문자열은 ASCII(`streamlit_app.py`의 plot lock).
- Streamlit 1.54 기준(`st.components.v2`, `st.popover`, `segmented_control`). 버전을 올리면 ui_audit을 다시 측정함.
- 화면 측정 도구가 의존하는 DOM 계약은 [UI 허브의 ui_audit DOM 계약](../docs/ui_redesign/README.md)을 따름. 바꾸면 도구도 같은 커밋에서 고침.
- 결정은 `docs/ui_redesign/decisions.md`에 새 항목으로 남김(번복도 새 항목). 작업 기록은 `log.md` 맨 위에.

## 검증

1. 관련 테스트: `python -m pytest -q tests/test_ui_*.py tests/test_sabes_*.py`와 바꾼 컴포넌트의 테스트.
2. 화면이 바뀌면 `python tools/ui_audit.py --tag <단계>` → `--compare baseline <단계>`, 스크린샷 확인.
3. 앱 확인: `.claude/launch.json`의 `gabes` 설정(`streamlit run streamlit_app.py`). SABES는 `?app=sabes`.
