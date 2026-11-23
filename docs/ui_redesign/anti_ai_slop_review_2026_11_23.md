# ⟦AI⟧ Gemini → Claude 인계 문서: GABES Anti-AI-Slop 정보량 최적화 및 전역 디자인 규약 개정안

> 출처: Gemini 독립 디자인 검수. 사용자가 2026-11-23 인계했다. 규칙 R1–R6는 제안이며 아직 채택되지 않았다.
> 진행은 [체크리스트](../checklist.json)의 `ui-anti-ai-slop-information-hierarchy`에서 관리한다.
> ⟦AI⟧ 보관 시 바꾼 점: 개인 Drive 링크를 저장소 상대 경로로 바꿨다. 원문이 언급한 화면 캡처는 포함하지 않았다.
> 원문이 예로 든 결정 번호 `D18`은 이미 쓰이고 있으므로, 채택하면 다음 번호(현재 D22)를 쓴다.

이 문서는 패치가 누적되며 발생한 **AI 특유의 과잉 정보 노출(방어적 문구 상시 출력, 동일 정보 다중 중복, 평탄·역전된 강조 계층)**을 정리하기 위한 **GABES Anti-AI-Slop 패치 개선안**입니다. 단발성 화면 수정에 그치지 않고, **모든 스킴의 시각적 캡처 감사**와 **기존 디자인 규약 문서 개정**을 통해 GABES 전체에 공통으로 강제되는 디자인 계약으로 정착시키는 것이 목표입니다.

## 1. 작업 목표 및 원칙

* **핵심 진단**: 현재 UI는 물리적 정확성을 증명하려는 방어적 설명문, 이미 상단에서 선택한 값의 반복 출력, 라벨만 굵고 정작 수치는 얇은 역전된 타이포그래피로 인해 인간의 시각 탐색을 방해함.
* **목표 수준**: 기능을 덜어내는 과도한 단순화가 아니라, **전문 계측기 수준의 정보 밀도와 명확한 시선 유도(Visual Hierarchy)**를 확보하는 것.
* **적용 범위**: `Rydberg-EIT` 우선 실증 → `OD`, `SAS`, `FWM` 등 GABES 전 스킴 시각 캡처 감사 → [gabes_ui/DESIGN.md](../../gabes_ui/DESIGN.md) 및 [docs/ui_redesign/decisions.md](decisions.md) 규약 개정.

## 2. 전역 Anti-AI-Slop 디자인 규약 개정 초안 (D4·D7 확장)

기존 [gabes_ui/DESIGN.md](../../gabes_ui/DESIGN.md)의 감량 원칙(D4)과 숫자 표시 규칙(D7)을 아래 6개 **전역 강제 규칙**으로 구체화하여 `decisions.md`에 새 결정 항목으로 추가하고 `DESIGN.md`의 `## 규칙` 섹션에 반영합니다.

| 규칙 ID | 원칙명 | 금지 사항 (AI Slop 패턴) | 강제 기준 (계측기 UX) |
| :--- | :--- | :--- | :--- |
| **R1** | **무문장 기본 화면 (Zero-Prose Canvas)** | 차트·테이블·Readout 아래 방어적 설명문·면책 문구 상시 출력 | 정상 상태에서 본문 문장 수 0개 지향. 정의·계산 방식은 툴팁/`Guide`로 이관하고, 경고는 예외 상황(결측·클램프 등) 발생 시에만 조건부 표시 |
| **R2** | **단일 출처 표시 (No Echo)** | 상단에서 이미 선택한 물리량(`Observable`)·조건·커서 값을 테이블 열이나 차트 y축 제목에 반복 출력 | 전역 선택값과 동일한 열·라벨은 생략. 테이블의 보조 좌표 열(`Sample`)은 행마다 값이 실제로 다를 때만 표시 |
| **R3** | **수치 우선 강조 (Value-First Hierarchy)** | 라벨·체크박스·보조 캡션까지 모두 Bold 처리하고 정작 물리 수치는 일반 굵기로 출력 | Bold는 **섹션 제목과 핵심 물리 수치(`tabular-nums`)**에만 허용. 라벨과 보조 옵션은 작고 옅은 `--g-muted` 톤 유지 |
| **R4** | **유효숫자 절제 (Display Precision)** | `-0 MHz`, `0.3375000000`, `0.894297`처럼 부동소수점 원시 정밀도나 음의 0 노출 | `format.py` 기준 화면 표시는 3~4자리 유효숫자 + `-0` 정규화 적용. 전체 정밀도는 호버 툴팁·복사·내보내기에서 제공 |
| **R5** | **1차 vs 2차 정보 분리 (Progressive Disclosure)** | `(display only)` 참고값이나 정적 셀 환경 상수를 핵심 제어·관측량과 동일 층위에 평면 나열 | 기본 영역에는 현재 스캔/비교에 직접 반응하는 핵심 파라미터·관측량 3~5개만 노출. 참고값·정적 환경 상수는 `Advanced` 또는 접이식 `Cell state` 스트립으로 분리 |
| **R6** | **단일 주 뷰 원칙 (No Vertical Stack Duplication)** | 같은 물리량의 전체 스펙트럼·히트맵·확대 곡선을 세로로 3단 누적 배치 | `[Curves \| Map]` 토글이 중단 주 뷰 1개를 전환하고, 단일 조건 상세 곡선은 전용 탭(`Single-condition details`)으로 분리 |

## 3. 기준 사례 감사: `Rydberg-EIT` (`?scheme=rydberg_eit`) 구체 개선 목록

첨부된 `Model spectra` 화면 캡처를 기준으로 위 6개 규칙을 즉시 적용할 항목입니다.

### 3.1 과감히 삭제 / 툴팁 이관
- [ ] **방어적 설명문 4곳 제거**:
  - 조건 맵 하단: `Each band is a calculated condition. Blank cells are missing, outside the native scan, or not calculated.` → 결측 셀이 실제로 존재할 때만 조건부 경고 표시.
  - 우측 테이블 하단: `Each curve retains its own solved grid. Rows report that condition's actual sampled coordinate.` → 테이블 헤더 툴팁 또는 `Guide`로 이관.
  - `Condition readouts` 부제: `Derived from each complete scan; independent of the inspection cursor.` → 섹션 툴팁으로 이관.
  - Readout 내부 2개 정의문(`Native interpolated crossings of...`, `Incident probe power × transmission contrast...`) → 각 항목 라벨 호버 툴팁으로 이관.
- [ ] **우측 테이블 중복 열 정리**: `Observable` 열(`Transmission T` 7행 반복) 제거. `Sample (MHz)` 열은 모든 조건의 샘플 좌표가 동일할 때 숨기고, 차이가 생기는 경우에만 표시.
- [ ] **차트 제목 / y축 이중 표기 제거**: `EIT contrast`, `EIT FWHM`, `Transmitted feature amplitude` 소형 차트와 `Transmission spectra` 상단의 중복 y축 제목을 제거하고 차트 제목에 단위 통합(예: `EIT FWHM [MHz]`).
- [ ] **부호 및 소수점 노이즈 수정**: 부제목의 `at -0 MHz` 정규화(`0 MHz`), `Selected sample`의 `0.3375 MHz` 텍스트 + `0.3375000000` 입력창 이중 구조를 편집 가능한 단일 필드로 통합.

### 3.2 계층 재정렬 및 압축
- [ ] **상단 제어부 압축**: `Microwave frequency (display only) · 37 GHz`와 기본값이 `0 MHz`인 비활성 디튜닝 알약을 `Advanced`로 이동해 `Conditions`를 1줄로 정리. 화면 전폭을 쓰는 `Compare` 드롭다운과 중복된 `Probe power (µW)`/`Selected condition` 입력을 컴팩트한 1줄 컨트롤 바로 통합.
- [ ] **우측 `Condition readouts` 타이포 역전 해소 및 2단 분리**: 라벨을 `--g-muted` 소형 텍스트로, 수치를 굵은 `tabular-nums` 대형 텍스트로 뒤집음. 핵심 관측량(`EIT FWHM`, `Contrast`, `Amplitude`, `Resonance T`)과 정적 셀 상수(`Vapor density`, `Effective atom number`, `Effective vapor temperature`)를 분리(셀 상수는 하단 접이식 `Cell state` 또는 1줄 메타 스트립).
- [ ] **세로 3단 `Transmission T` 뷰 정리**: `[Curves | Map]`이 메인 비교 뷰 1개를 전환하도록 통합하고, 하단 단일 조건 확대 플롯은 `Single-condition details` 탭과 역할을 분리해 스크롤 길이 축소.
- [ ] **보조 체크박스 톤 다운**: `Show residual-width comparison`, `Show assumed opacity scenarios`의 Bold를 제거하고 보조 톤(또는 `Display options` 팝오버)으로 정리.

## 4. 전 스킴 시각 캡처 감사 및 규약 개정 실행 절차 (Claude 체크리스트)

### Phase 1 — 전 스킴 베이스라인 캡처 및 감사
- [ ] `.claude/launch.json`의 `gabes` 설정으로 앱을 실행하고 `python -m gabes_lab.tools.ui_audit --tag pre_anti_slop` 실행.
- [ ] 아래 모든 스킴의 실제 화면을 시각적으로 캡처해 **R1~R6 위반 목록**을 스킴별 표로 작성:
  - `Rydberg-EIT` (`EIT`, `AT electrometry` 양쪽 모드 + `Model spectra` / `Single-condition details`)
  - `OD` (Optical Depth / 실험 CSV 비교 포함)
  - `SAS` (Saturated Absorption Spectroscopy)
  - `FWM` (Four-Wave Mixing)
  - 기타 활성 스킴 및 `?app=sabes` (SABES 규칙과의 충돌 여부만 확인)
- [ ] 각 스킴에서 **(a) 상시 노출 문장 수, (b) 첫 그래프 시작 Y좌표(above-the-fold), (c) 중복 열/라벨 수, (d) Bold 적용 요소 목록**을 정량 기록.

### Phase 2 — 디자인 규약 문서 개정
- [ ] [docs/ui_redesign/decisions.md](decisions.md)에 **Anti-AI-Slop 및 정보 계층 규약(R1~R6)**을 신규 결정 항목(예: `D18`)으로 추가.
- [ ] [gabes_ui/DESIGN.md](../../gabes_ui/DESIGN.md)의 `## 규칙` 섹션에 R1~R6 요약과 `ui_audit` 검증 기준을 명문화.
- [ ] `docs/ui_redesign/log.md` 최상단에 전 스킴 감사 결과와 규약 개정 이력 기록.

### Phase 3 — 공통 컴포넌트 우선 패치 및 스킴별 적용
- [ ] 공통 컴포넌트(`readout.py`, `spectral_compare.py`, `model_compare_section.py`, `conditions.py`, `format.py`, `gabes.css`)를 먼저 수정해 개별 스킴 코드 중복 없이 R1~R6가 전역 자동 적용되도록 구현.
- [ ] `Rydberg-EIT`를 포함한 전 스킴의 개별 오버라이드·면책 문구 정리.
- [ ] `ui_audit` DOM 계약([docs/ui_redesign/README.md](README.md)) 변경이 필요한 경우 측정 도구도 같은 커밋에서 동기화.

### Phase 4 — 회귀 검증 및 Before/After 시각 검수
- [ ] `python -m pytest -q tests/test_ui_*.py tests/test_sabes_*.py` 전체 통과 확인.
- [ ] `python -m gabes_lab.tools.ui_audit --tag post_anti_slop` 실행 후 `--compare pre_anti_slop post_anti_slop` 비교.
- [ ] 전 스킴 Before/After 스크린샷을 직접 육안 검수하여 **(1) 물리 정보 손실 없음, (2) 첫 그래프 가시성 개선, (3) 시선이 수치와 곡선에 먼저 꽂히는지** 최종 확인.
