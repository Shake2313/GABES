# ⟦AI⟧ 정보 계층 감사 1단계 · R1–R6 위반 목록 (pre_anti_slop)

> [Gemini 검수](../../anti_ai_slop_review_2026_11_23.md)의 1단계, 읽기 전용. 체크리스트 `ui-anti-ai-slop-information-hierarchy`.
> R1–R6는 아직 채택 전 제안. 이 표는 위반 후보 기록이며 수정 지시 아님. 2–4단계는 OD/SAS AutoOD 통합 마감 뒤.

## 범위·방법

- 커밋 `3925303`(로컬 미커밋 `gabes/fwm_zeeman_response.py`는 화면 경로 무관), Streamlit 1.54, headless Edge 1440×900, 2026-11-24.
- 기준 측정: `python -m gabes_lab.tools.ui_audit --tag pre_anti_slop` → 같은 폴더 [summary.md](summary.md)·`metrics.json`.
- 화면별 인벤토리: `runs/ui_audit/anti_slop_phase1/probe_views.py`(ui_audit CDP 재사용, 기본값에서 각 화면을 새 세션으로 열고 텍스트·굵기·입력값·그래프·표 수집, 900 px 뷰포트 스크롤 이어붙인 전체 캡처) → `analyze.py` → `build_tables.py`. 산출물은 `runs/`(git 밖, 아카이브 대상): `runs/ui_audit/anti_slop_phase1/20261124T072323Z-3925303/`.
- 화면 17개: OD/SAS 3, Λ 4, Rydberg 4(EIT·AT × Model spectra·Single-condition details), Magneto 2, FWM 2, PSR 2. SABES는 규칙 충돌만 문서로 검토(캡처 없음).
- 집계 규칙
  - (a) 상시 문장: 본문·사이드바에서 위젯 라벨·버튼·제목·접힘 제목·그래프 SVG 글자·표 칸·경고 상자를 뺀 블록 중 5단어 이상이고 문장부호(. ! ?)로 끝나는 문장 수. 닫힌 `<details>`·1–2 px 스크린리더 텍스트·조상에 잘린 요소 제외. 같은 문장이 그래프마다 반복되면 각각 셈.
  - (b) 첫 그래프 Y: 페이지 맨 위에서 본문 첫 그래프(img·canvas·svg, 면적 > 40,000 px²)의 상단. ui_audit의 "plot top"은 가장 큰 그래프 기준이라 값이 다를 수 있음.
  - (c) 중복 라벨: 제목·위젯 라벨·짧은 캡션·표 머리·그래프 글자를 소문자·괄호 단위 제거 후 비교, 여분 복사본 수. 상수 열: 데이터 3행 이상 모두 같은 표 열.
  - (d) 굵게(font-weight ≥ 600): 위젯 라벨 / 그 밖(그래프 제목·readout 라벨·섹션명) / 숫자 값. 제목 분류는 사람이 표에서 다시 판단.
  - R4 후보: 화면 숫자 중 유효숫자 5자리 이상 소수, 음의 0, 입력칸 부동소수 꼬리.
- 한계: 휴리스틱 집계. matplotlib 그림(img)의 축 글자는 읽지 못해 (c)에서 빠짐. 사이드바가 있는 FWM 전체 캡처는 이어붙이기에서 사이드바가 겹쳐 보임(측정값 무관).

## 1. 정량표

| 화면 | (a) 상시 문장 | (b) 첫 그래프 Y (px) | 그래프 수 | (c) 중복 라벨 | (c) 상수 열 | (d) 굵게: 위젯 라벨 / 그 밖 / 값 | R4 후보 |
|---|---:|---:|---:|---:|---:|---|---:|
| OD/SAS · OD 프리셋 | 1 | 376 | 3 | 8 | 0 | 5 / 3 / 4 | 4 |
| OD/SAS · SAS 기본 | 1 | 376 | 3 | 7 | 0 | 5 / 3 / 2 | 2 |
| OD/SAS · Fit OD data + CSV | 3 | 637 | 3 | 6 | 0 | 3 / 3 / 4 | 11 |
| Λ · EIT Model spectra | 12 | 617 | 7 | 18 | 2 | 7 / 6 / 0 | 26 |
| Λ · AT Model spectra | 7 | 617 | 6 | 17 | 2 | 7 / 5 / 0 | 22 |
| Λ · CPT Model spectra | 11 | 617 | 6 | 19 | 2 | 6 / 6 / 0 | 8 |
| Λ · EIT Single-condition | 0 | 476 | 1 | 0 | 0 | 0 / 5 / 0 | 0 |
| Rydberg · EIT Model spectra | 13 | 679 | 7 | 20 | 2 | 9 / 6 / 0 | 16 |
| Rydberg · AT Model spectra | 6 | 703 | 6 | 15 | 2 | 8 / 5 / 0 | 12 |
| Rydberg · EIT Single-condition | 0 | 517 | 1 | 0 | 0 | 0 / 6 / 0 | 0 |
| Rydberg · AT Single-condition | 0 | 546 | 1 | 0 | 0 | 0 / 7 / 0 | 2 |
| Magneto · Model spectra | 9 | 648 | 4 | 12 | 2 | 8 / 4 / 0 | 6 |
| Magneto · Single-condition | 0 | 476 | 1 | 0 | 0 | 0 / 3 / 0 | 0 |
| FWM · Squeezing | 0 | 479 | 2 | 0 | 0 | 0 / 12 / 0 | 2 |
| FWM · Biphoton | 0 | 486 | 1 | 0 | 0 | 0 / 14 / 0 | 1 |
| PSR · Model spectra | 3 | 574 | 3 | 10 | 2 | 5 / 3 / 0 | 6 |
| PSR · Single-condition | 0 | 592 | 1 | 0 | 0 | 0 / 8 / 0 | 0 |

ui_audit 기본 화면 지표(같은 커밋): 가장 큰 그래프 상단 데스크톱 sas 376 · fwm 479 · psr 574 · magneto 648 · lambda 836 · rydberg_eit 899 px, 휴대폰 584–1223 px. [summary.md](summary.md).

## 2. 스킴별 R1–R6 위반 후보

| 스킴 | R1 상시 설명문 | R2 중복 표시 | R3 강조 역전 | R4 표시 정밀도 | R5 1차/2차 혼재 | R6 세로 중복 뷰 |
|---|---|---|---|---|---|---|
| OD/SAS Simulate | 1문장("Frequency reference…"). 그 밖 정의는 접힌 Model details 안 → 대체로 준수 | 그래프 제목+축 제목 "Transmission"/"Transmission T", "Optical depth (OD)"/"Optical depth τ" | 판독부는 값 굵게·라벨 보통(준수). 위젯 라벨 7개 굵게 | "+1.10614 MHz", "79.2091%", "0.233078", "7,115,000,000"(→7.115×10⁹), 표본 주파수 입력 "-2586.01983724" | 문제 적음 | 전체 + 확대 T·OD 두 개: 확대는 다른 물리량 병치라 해당 약함 |
| OD/SAS Fit OD + CSV | 3문장: "The native frequency scale is preserved. Confirm its calibration…"(2) + 정규화 요약 1줄 | 개요·Detail 모두 "Transmission / relative signal" 축 | 판독부 준수 | 창 입력 "-0.57978739", "2.240218047", "0.830215326", "0.83021533 GHz", "99.5839%" | Experimental CSV 패널이 첫 화면 아래에서 길게 이어짐 | 개요 T + Detail T: 같은 양(확대 목적) |
| Λ EIT/AT/CPT | 7–12문장: 지도 캡션, "Each curve retains…", "Derived from…", 소형 그래프·readout 정의문이 두 번씩 반복(“Both half-height crossings…”, “T(δ=0) − min T…”), "Spectral quality indicator…", AT의 "…not an EIT/ATS classifier." | 표 Observable 열 "Transmission T" ×9·Sample 열 0 고정. 소형 그래프 제목+축+범례 3중("Transparency contrast ΔT"). 조건 라벨이 범례·드롭다운·표에 반복 | readout 라벨 굵게·값 보통(역전). "Show all observables" 굵게 | 표 "1.14075e−6", "0.000292438"; 지도 색 막대 최소 "1.4564e−14"(AT "1.1456e−15", 부동소수 잡음); 축 눈금 "2.8284", "1.4142"; 범례 "0.353553 mW" 등 √2 격자 6자리 | 판독 7개 평면 나열 | Full range 곡선 + 조건 지도 + 확대 T가 세로로 쌓임. Map 선택해도 곡선 유지 |
| Rydberg EIT/AT | EIT 13문장(Gemini 지적 4곳 확인, 소형 그래프 캡션 2문장 × 3회 포함), AT 6문장 | Observable 열 ×7·Sample 열 0; 소형 그래프 제목+축+범례 3중; 부제 "Probe detuning [MHz] · …" | readout 라벨 굵게·값 보통; 체크박스 2개 굵게 | 부제 "at -0 MHz"(EIT·AT); Gemini가 본 "0.3375 MHz" 표본·입력 이중 표기는 기본 표본 0 MHz에서는 재현 안 됨; "154.171 mV/m", "7.54262e+15 m⁻³"; 접힌 창 입력 "-5.550000000000001" | "Microwave frequency (display only) · 37 GHz" 알약이 조건 줄에 노출; 셀 상수(밀도·N_eff·온도 3종)가 핵심 관측량과 같은 층 | Λ와 같은 3단 적층 |
| Magneto | 9문장: "Rotation is defined only for linear input…"(미지원 사유, R1 보호 대상), "Vertical offsets only…"(2), readout 정의 2개, 공통 비교 문장 3 | Observable "Rotation" ×6·Sample 0; 축 "Commanded longitudinal field (µT)" ×4 + 입력 라벨; "Bcmd = 0 / Bz = 0" 표식 ×4 | readout 역전 | B=0 회전 "−1.69980e−14 mrad"·"−1.69980e−17 rad"(대칭상 0인 값의 부동소수 잡음을 값으로 표시), "0.000890707" | 조건 알약 6개 | Full range 회전 + 확대 회전 + 투과 오프셋 |
| Magneto Single-condition | 0문장(범위 고지 "Conditional model; …"는 접힌 영역 안) | — | readout 라벨 11개 굵게 | — | — | 단일 그림 |
| FWM Squeezing/Biphoton | 0문장(사이드바 도움말은 접힌 항목 안) | 그림이 img라 자동 판정 불가; 육안상 중복 적음 | 핵심 값이 큰 글자로 먼저 보임(가까운 모범). 라벨도 굵게 | 사이드바 "0.000 ×10⁻¹⁴ Hz⁻¹" 표시 | 사이드바 섹션 구분 명확 | 이득·잡음 두 패널은 다른 양 |
| PSR | 3문장(공통 비교 문장) | Observable "Noise / shot noise" ×3·Sample 0; "Full range · Stokes A" + "Stokes A" | readout 역전 | "0.54569 dB", "−1.37824 dB", "129.303 °", "0.751468" | 조건 알약 9개(2줄) | Full range Stokes A + 확대 Stokes A |
| Single-condition(Λ·Rydberg·PSR) | 0문장 | — | readout 라벨 5–8개 굵게, 값 보통 | AT "154.171 mV/m RMS" | — | 단일 matplotlib 그림 |

### 공통 원인 (3단계 지렛대)

- Λ·Rydberg·Magneto·PSR 네 스킴의 Model spectra는 같은 비교 컴포넌트를 씀. 같은 문장 블록("Each band…"(지도 표시 때), "Each curve retains…", "Derived from each complete scan…"), Sample/Observable 상수 열, 제목+축 중복, readout 라벨 굵게·값 보통, Full range·지도·확대 3단 적층이 모두 여기서 나옴. 공통 컴포넌트 한 곳을 고치면 네 스킴에 함께 적용됨.
- OD/SAS 판독부(값 굵게, 라벨 보통, 단위 분리)는 이미 R3 방향. D21 AutoOD 계열 규약의 참조 구현 후보.
- 첫 그래프가 600–700 px까지 밀리는 이유: 조건 알약 줄 + 전폭 Compare 선택 + 전폭 입력 + Condition scan 버튼 + 섹션 제목·부제·범례 + Curves/Map·Quantity 줄 + 체크박스 줄. OD/SAS는 같은 자리에 376 px.
- 기존 D4("plot 제목: 앱 화면에선 숨김", "plot 제목(파라미터 반복) = rail 값")와 이미 어긋나는 부분: 비교 화면의 그래프 제목+축 제목 이중 표기. R2는 D4 재확인 성격.

### R1 보호 대상 (조건부, 유지)

"resolution-limited"(Λ FWHM), "PREDICTED · opacity uncalibrated"(Rydberg 배지), "unsupported polarization"·"Rotation is defined only for linear input…"(Magneto 미지원 사유), "Conditional estimate · within approximation screens"(PSR), FWM 보정 외삽 표시. 체크리스트 overlap_guards와 같음.

## 3. SABES 충돌 검토 (문서 기준)

| 규칙 | SABES 근거 | 판정 |
|---|---|---|
| R1 | D13·[SABES 디자인 규칙](../../../SABES/design_rules.md): `*` 뜻을 범례에 한 번 표시(사용자 요구). D19: 모의 신호 안내 유지 | 충돌. 채택 시 R1 예외로 "사용자 요구 범례·모의 신호 안내" 명시 필요 |
| R2 | 소자마다 `(lumped)` 반복 금지(이미 시행) | 일치 |
| R3 | 광선 식별색·대비 규칙 고정, 요약 카드는 GABES 읽기값 띠 재사용(D19) | 굵기만 공통 띠에서 바뀌면 일치. 색·명도는 D21 사용자 결정 |
| R4 | D19: 요약 카드는 SABES의 숫자·단위 문자열을 그대로 사용 | 부분 충돌. `format.py` 일괄 규칙이 SABES 문자열에 닿지 않음 → 적용 범위 결정 필요 |
| R5 | 상단 요약 3개 카드(사용자 채택) | 일치(3–5개 범위) |
| R6 | 탭 구조, 세로 중복 뷰 없음 | 해당 없음 |

## 4. 감사 중 발견한 결함 (R1–R6 밖)

- Fit OD data + CSV(로컬, 새 세션에서 `?scheme=sas&compare=0`로 바로 열고 업로드): 노란 경고 "The widget with key "sas___csv_auto_correct" was created with a default value but also had its value set via the Session State API." 배포앱(Simulate → Fit 전환 경로)에서는 안 보였음. Codex 레인 전달.
- Magneto B=0 회전값이 0 대신 ±1.7×10⁻¹⁴ mrad로 표시됨. 계산이 아닌 표시 정밀도 문제로 보임(R4 적용 시 해결 대상).
- 확대 창 From/To 입력의 부동소수 꼬리(`-5.550000000000001`, `±4.008735581642769`, `-3173.5515794999997`). 창을 펼칠 때 보임.

## 부록 · 화면별 목록

#### OD/SAS · OD 프리셋

- 상시 문장: (1) Frequency reference: Rb85 D1 line centroid.
- 중복 라벨: transmission t ×5, optical depth ×2, optical depth τ ×2, relative frequency ×3
- 굵은 위젯 라벨: Doppler group · Pump power (mW) · Probe power (µW) · Absorption peak · Sample frequency (MHz)
- 굵은 그 밖 텍스트: Full transmission spectrum · Transmission · Optical depth (OD) · Conditions · Optical depth · Model details
- 굵은 값: 0.233078 · 0.233078 · 7,115,000,000 · 79.2091%
- R4 후보: 0.233078, 79.2091, -2586.01983724 (input)

#### OD/SAS · SAS 기본

- 상시 문장: (1) Frequency reference: Rb85 D1 line centroid.
- 중복 라벨: transmission t ×5, optical depth τ ×2, relative frequency ×3
- 굵은 위젯 라벨: Doppler group · Pump power (mW) · Probe power (µW) · Model reference · Sample frequency (MHz)
- 굵은 그 밖 텍스트: Full transmission spectrum · Transmission · Optical depth (OD) · Conditions · SAS detuning · Model details
- 굵은 값: +1.10614 MHz · +16.21 pp
- R4 후보: +1.10614, -2586.01983724 (input)

#### OD/SAS · Fit OD data + CSV

- 상시 문장: (2) The native frequency scale is preserved. Confirm its calibration before a quantitative fit. ‖ (1) Relative normalized transmission [Arb. unit]: relative floor 0.513155 · relative ceiling 1.99725 · smoothing window 7
- 경고 상자: The widget with key "sas___csv_auto_correct" was created with a default value but also had its value set via the Session State API.
- 중복 라벨: transmission / relative signal ×4, optical depth ×2, relative frequency ×3
- 굵은 위젯 라벨: Window start (GHz) · Window end (GHz) · Selected frequency (GHz)
- 굵은 그 밖 텍스트: Spectrum overview · Detail · transmission · Detail · optical depth · Conditions · Model at selected frequency · 0.83021533 GHz · Nearest measured sample
- 굵은 값: 0.00416921 · 0.00416921 · 7,115,000,000 · 99.5839%
- R4 후보: 0.83021533, 0.00416921, 99.5839, 0.820802, 0.995822, 0.513155, 1.99725, -0.579787394384 (input), 2.24021804738 (input), 0.8302153265 (input)

#### Λ · EIT Model spectra

- 상시 문장: (2) Each band is a calculated condition. Blank cells are missing, outside the native scan, or not calculated. ‖ (1) T(δ = 0) − min T of this complete native scan; absolute transmission difference. ‖ (2) Both half-height crossings must lie inside the scan. Fewer than six native intervals is approximate. ‖ (2) Each curve retains its own solved grid. Rows report that condition's actual sampled coordinate. ‖ (1) Derived from each complete scan; independent of the inspection cursor. ‖ (1) T(δ = 0) − min T of this complete native scan; absolute transmission difference. ‖ (2) Both half-height crossings must lie inside the scan. Fewer than six native intervals is approximate. ‖ (1) Spectral quality indicator; not clock stability or Allan deviation.
- 중복 라벨: coupling power ×5, transmission t ×4, re χ ×2, transparency contrast δt ×4, transparency-window fwhm ×4, probe detuning δp ×5
- 상수 열: Sample (MHz) = 0; Observable = Transmission T
- 굵은 위젯 라벨: Compare · Coupling power (mW) · Quantity · Selected condition · Map vertical axis · Probe detuning Δp (MHz) · Show all observables
- 굵은 그 밖 텍스트: Full range · Transmission · Transmission · condition map · Transmission · Dispersion from the same solve · Transparency contrast ΔT · Transparency-window FWHM · Conditions · EIT · Coupling power comparison · Selected sample · 0 MHz · Condition readouts · Effective Ωc/2π · Resonance transmission · Transparency contrast ΔT · Transparency-window FWHM · Contrast / FWHM · Samples per FWHM · Calculation conditions and scope
- 굵은 값: 없음
- R4 후보: 1.4564e−14, 0.33043, 1.14075e−6, 0.353553, 10.257, 2.16539e−5, 12.198, 0.000292438, 0.707107, 14.505, 0.00248428, 0.0130491, 1.41421, 20.514, 0.0448656, 24.395, 0.110016, 2.82843, 29.011, 0.209189, 0.330429, 0.15618, 0.0835514, 0.679044

#### Λ · AT Model spectra

- 상시 문장: (2) Each band is a calculated condition. Blank cells are missing, outside the native scan, or not calculated. ‖ (1) Two absorption maxima (transmission minima); central contrast must exceed 1%. ‖ (2) Each curve retains its own solved grid. Rows report that condition's actual sampled coordinate. ‖ (1) Derived from each complete scan; independent of the inspection cursor. ‖ (1) Two absorption maxima (transmission minima); central contrast must exceed 1%.
- 중복 라벨: transmission t ×4, effective ωc/2π ×4, re χ ×2, absorption-peak spacing ×4, probe detuning δp ×5, absorption peak 1 ×3, absorption peak 2 ×2
- 상수 열: Sample (MHz) = 0; Observable = Transmission T
- 굵은 위젯 라벨: Compare · Coupling power (mW) · Quantity · Selected condition · Map vertical axis · Probe detuning Δp (MHz) · Show all observables
- 굵은 그 밖 텍스트: Full range · Transmission · Transmission · condition map · Transmission · Dispersion from the same solve · Absorption-peak spacing · Conditions · AT · Coupling power comparison · Selected sample · 0 MHz · Condition readouts · Effective Ωc/2π · Resonance transmission · Absorption-peak spacing · Calculation conditions and scope
- 굵은 값: 없음
- R4 후보: 1.1456e−15, 0.99726, 0.957137, 0.353553, 27.352, 0.969486, 32.527, 0.97832, 0.707107, 38.681, 0.984618, 0.989097, 1.41421, 54.704, 0.992278, 65.054, 0.994533, 2.82843, 77.362, 0.996131, 0.997262

#### Λ · CPT Model spectra

- 상시 문장: (1) T(δ = 0) − min T of this complete native scan; absolute transmission difference. ‖ (2) Both half-height crossings must lie inside the scan. Fewer than six native intervals is approximate. ‖ (1) Spectral quality indicator; not clock stability or Allan deviation. ‖ (2) Each curve retains its own solved grid. Rows report that condition's actual sampled coordinate. ‖ (1) Derived from each complete scan; independent of the inspection cursor. ‖ (1) T(δ = 0) − min T of this complete native scan; absolute transmission difference. ‖ (2) Both half-height crossings must lie inside the scan. Fewer than six native intervals is approximate. ‖ (1) Spectral quality indicator; not clock stability or Allan deviation.
- 중복 라벨: coupling power ×4, transmission t ×4, re χ ×2, transparency contrast δt ×4, transparency-window fwhm ×4, contrast / fwhm ×4, two-photon detuning δp − δc ×4
- 상수 열: Sample (kHz) = 0; Observable = Transmission T
- 굵은 위젯 라벨: Compare · Coupling power (mW) · Quantity · Selected condition · Two-photon detuning Δp − Δc (kHz) · Show all observables
- 굵은 그 밖 텍스트: Full range · Transmission · Transmission · Dispersion from the same solve · Transparency contrast ΔT · Transparency-window FWHM · Contrast / FWHM · Conditions · CPT · Coupling power comparison · Selected sample · 0 kHz · Condition readouts · Effective Ωc/2π · Resonance transmission · Transparency contrast ΔT · Transparency-window FWHM · Contrast / FWHM · Samples per FWHM · Calculation conditions and scope
- 굵은 값: 없음
- R4 후보: 0.259331, 0.706497, 0.916211, 804.614, 0.000878056, 3.48092

#### Λ · EIT Single-condition

- 상시 문장: 없음
- 중복 라벨: 없음
- 굵은 위젯 라벨: 없음
- 굵은 그 밖 텍스트: Transmission at resonance · Window FWHM · Group index n_g · Group delay · Not assessed · Conditions
- 굵은 값: 없음
- R4 후보: 없음

#### Rydberg · EIT Model spectra

- 상시 문장: (2) Each band is a calculated condition. Blank cells are missing, outside the native scan, or not calculated. ‖ (2) Native ±5 MHz local-floor feature. Opacity scenarios are assumptions; optima are sampled model results. ‖ (2) Native ±5 MHz local-floor feature. Opacity scenarios are assumptions; optima are sampled model results. ‖ (2) Native ±5 MHz local-floor feature. Opacity scenarios are assumptions; optima are sampled model results. ‖ (2) Each curve retains its own solved grid. Rows report that condition's actual sampled coordinate. ‖ (1) Derived from each complete scan; independent of the inspection cursor. ‖ (1) Native interpolated crossings of the center-to-local-floor feature within ±5 MHz. ‖ (1) Incident probe power × transmission contrast; excludes path/detector efficiency.
- 중복 라벨: probe power ×6, transmission t ×4, eit contrast ×3, eit fwhm ×4, transmitted feature amplitude ×4, probe detuning ×4, bare resonance ×2
- 상수 열: Sample (MHz) = 0; Observable = Transmission T
- 굵은 위젯 라벨: Compare · Probe power (µW) · Coupling power (mW) · Quantity · Selected condition · Show residual-width comparison · Show assumed opacity scenarios · Probe detuning (MHz) · Show all observables
- 굵은 그 밖 텍스트: Full range · Transmission spectra · Transmission spectra · condition map · Transmission spectra · EIT contrast · EIT FWHM · Transmitted feature amplitude · Conditions · EIT · Probe power comparison · Selected sample · 0 MHz · Condition readouts · EIT FWHM · Center-to-local-floor contrast · Transmitted feature amplitude · Resonance transmission · Vapor density · Effective atom number · Effective vapor temperature · Cold-spot temperature · Heater setpoint · Calculation conditions and scope
- 굵은 값: 없음
- R4 후보: -0 (negzero), 0.74458, 0.96661, 0.920138, 0.922498, 0.926808, 0.934089, 0.940001, 0.944899, 0.949022, 1.60349, 0.136242, 0.817455, 7.54262e+15, 1.66611e+6

#### Rydberg · AT Model spectra

- 상시 문장: (2) Each band is a calculated condition. Blank cells are missing, outside the native scan, or not calculated. ‖ (2) Each curve retains its own solved grid. Rows report that condition's actual sampled coordinate. ‖ (1) Derived from each complete scan; independent of the inspection cursor. ‖ (1) Ideal two-level conversion; optical line pulling is not corrected.
- 중복 라벨: microwave rabi ω_rf ×5, transmission t ×4, electric field ×2, optical transmission-peak separation ×4, probe detuning ×4, transmission peak 1 ×2
- 상수 열: Sample (MHz) = 0; Observable = Transmission T
- 굵은 위젯 라벨: Compare · Probe power (µW) · Coupling power (mW) · Microwave Rabi Ω_RF (MHz) · Quantity · Selected condition · Probe detuning (MHz) · Show all observables
- 굵은 그 밖 텍스트: Full range · Transmission spectra · Transmission spectra · condition map · Transmission spectra · Input field and peak-based estimate · Optical transmission-peak separation · Conditions · AT electrometry · Microwave Rabi Ω_RF comparison · Selected sample · 0 MHz · Condition readouts · Optical transmission-peak separation · Field from input ΩRF · Field estimated from optical peak separation · Bare-resonance transmission · Effective vapor temperature · Cold-spot temperature · Heater setpoint · Vapor density · Calculation conditions and scope
- 굵은 값: 없음
- R4 후보: -0 (negzero), 0.79625, 0.99396, 0.940001, 0.933396, 0.839903, 0.796247, 3.16976, 154.171, 132.077, 7.54262e+15

#### Rydberg · EIT Single-condition

- 상시 문장: 없음
- 중복 라벨: 없음
- 굵은 위젯 라벨: 없음
- 굵은 그 밖 텍스트: EIT linewidth · Transmission at resonance · Max spectral slope · IF discriminator · IF optimum detuning · PREDICTED · opacity uncalibrated · Conditions
- 굵은 값: 없음
- R4 후보: 없음

#### Rydberg · AT Single-condition

- 상시 문장: 없음
- 중복 라벨: 없음
- 굵은 위젯 라벨: 없음
- 굵은 그 밖 텍스트: Total sensitivity · PSN-limited sensitivity · RF field from fitted Ω_RF · RF AT splitting · AT split field estimate · AT center shift · PREDICTED · opacity uncalibrated · Conditions
- 굵은 값: 없음
- R4 후보: 154.171, 132.077

#### Magneto · Model spectra

- 상시 문장: (2) Rotation is defined only for linear input in this model. Elliptical and circular input curves are unavailable; compare their transmission. ‖ (2) Vertical offsets only; all curves share the same transmission scale. Hover and the readout table show the original values. ‖ (2) Each curve retains its own solved grid. Rows report that condition's actual sampled coordinate. ‖ (1) Derived from each complete scan; independent of the inspection cursor. ‖ (1) A signed sampled contrast; check the physical field and scan extent. ‖ (1) Selected signal's photon shot-noise limit; atomic and technical noise are excluded.
- 중복 라벨: rotation ×4, field asd · log ×2, qwp angle · transmission ×2, commanded longitudinal field ×5, bcmd = 0 / bz = 0 ×4
- 상수 열: Sample (µT) = 0; Observable = Rotation
- 굵은 위젯 라벨: Compare · QWP angle (°) · Beam intensity (mW/cm²) · Physical longitudinal bias (µT) · Quantity · Selected condition · Commanded longitudinal field (µT) · Show all observables
- 굵은 그 밖 텍스트: Full range · Polarization rotation · Polarization rotation · Photon shot-noise magnetic sensitivity · Transmission across incident polarizations · Conditions · Magnetic field · condition comparison · Selected sample · 0 µT · Condition readouts · Rotation at command zero · Transmission at command zero · Mean transmission at scan endpoints · Transmission at command zero minus endpoint mean · Photon shot-noise field limit at command zero · Calculation conditions and scope
- 굵은 값: 없음
- R4 후보: −1.69980e−14, −1.69980e−17, 0.962035, 0.961145, 0.000890707, 8.77634

#### Magneto · Single-condition

- 상시 문장: 없음
- 중복 라벨: 없음
- 굵은 위젯 라벨: 없음
- 굵은 그 밖 텍스트: Rotation at declared bias · PSN magnetic sensitivity · Small-field SNR · Conditions
- 굵은 값: 없음
- R4 후보: 없음

#### FWM · Squeezing

- 상시 문장: 없음
- 중복 라벨: 없음
- 굵은 위젯 라벨: 없음
- 굵은 그 밖 텍스트: GABES · beta · DEFAULTS · Readout · Detunings · Beams · Detection & scaling · Intensity-difference noise · Seed gain G · s · Conjugate gain G · c · Conditions
- 굵은 값: 없음
- R4 후보: 6.2468, 74.7443

#### FWM · Biphoton

- 상시 문장: 없음
- 중복 라벨: 없음
- 굵은 위젯 라벨: 없음
- 굵은 그 밖 텍스트: GABES · beta · DEFAULTS · Model · Fields · Detunings · Phase matching · Detection & scaling · Peak signal–idler correlation · Pair rate · Model status · CAR · Intrinsic biphoton width · Detected biphoton width · Conditions
- 굵은 값: 없음
- R4 후보: 430.30

#### PSR · Model spectra

- 상시 문장: (2) Each curve retains its own solved grid. Rows report that condition's actual sampled coordinate. ‖ (1) Derived from each complete scan; independent of the inspection cursor.
- 중복 라벨: noise / shot noise ×6, analysis phase ×4, 0 db · shot noise ×3
- 상수 열: Sample (°) = 0; Observable = Noise / shot noise
- 굵은 위젯 라벨: Carrier intensity (mW/cm²) · Quantity · Selected condition · Analysis phase (°) · Show all observables
- 굵은 그 밖 텍스트: Full range · Stokes A · Stokes A · Orthogonal Stokes B · Conditions · Conditional squeezing · condition comparison · Selected sample · 0 ° · Condition readouts · Minimum conditional noise · Minimum-noise phase · Transmission at selected optical detuning · Calculation conditions and scope
- 굵은 값: 없음
- R4 후보: 0.54569, 1.36919, 2.06116, −1.37824, 129.303, 0.751468

#### PSR · Single-condition

- 상시 문장: 없음
- 중복 라벨: 없음
- 굵은 위젯 라벨: 없음
- 굵은 그 밖 텍스트: Minimum polarization noise · Maximum polarization noise · Optical detuning · Cell transmission · PSR shear κ · Minimum-noise phase · Conditional estimate · within approximation screens · Atomic excess noise omitted · Conditions
- 굵은 값: 없음
- R4 후보: 없음
