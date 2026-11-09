# ⁸⁵Rb D1 FWM 실험 문헌 DB

요청일: 2026-10-23. 범위: ⁸⁵Rb D1 double-Λ FWM의 intensity-difference squeezing 실험. 기존 [논문 CSV](../../archive/fwm_squeezing_paper_parameters.csv)의 10편 전부 추적, 원문 재검수·추가 문헌 조사. 이번 산출물은 DB. Gain 보정 패치·학습·성능 주장은 후속 작업.

⟦AI⟧ 2026-10-30: 사용자 요청으로 레거시 CSV를 `archive/`에 원본 그대로 보관. 현재 문헌 DB는 이 폴더. CSV는 과거 분석 재현·10편 이관 확인에만 사용. [이동·복구 기록](../../archive/fwm_squeezing_csv_migration_2026-10-30.json).

## 열람

- **[검색·비교 열람본](index.html)**: 논문 원제·포함 범위·운전점·수치·원문 위치 확인. 인터넷 없이 열람 가능. 원문 웹 링크에는 연결 필요.
- **[운전점 비교 CSV](experiments.csv)**: 실험 한 행. 같은 논문의 최적점·스캔·다른 장치 구성을 별도 기록.
- **[전체 수치 CSV](observations.csv)**: 수치 한 행. 정의·기준면·수치 유형·원문 위치·source SHA 포함.
- **[논문 목록](papers.csv)** / **[전체 JSON](database.json)** / **[논문별 원본 JSON](papers/)**.
- **[구조·검증 결과](validation.json)** / **[스키마](SCHEMA.md)**.

CSV는 Excel 호환 UTF-8 BOM. JSON은 UTF-8. 원본은 `papers/*.json`; CSV·HTML·결합 JSON은 재생성 가능한 읽기용 뷰. 기존 PDF는 원래 경로 참조, 새 원문은 `sources/` 저장. 원문 해시와 판본을 함께 기록. 수치 CSV의 `unit`은 기록 단위, `normalized_unit`은 µW/uW·µm/um·°C/degC·%/percent 표기 별칭을 통일한 코드. 값의 배율은 변경하지 않음.

## ⟦AI⟧ 기록·환산 규칙

`reported` 보고값, `approximate` 근삿값, `range` 구간, `lower_bound`/`upper_bound` 단측 경계, `derived` 환산값, `digitized` 그림 추출, `not_reported` 미보고. 미보고 값에 GABES 기본값을 채우지 않음. 오차가 미보고면 임의 오차막대도 추가하지 않음.

원문의 detuning·빔 크기는 정의 그대로 보존. 근거가 확실할 때만 별도 표준화:

\[
\Delta_\mathrm{GABES}=\nu_p-\nu(^{85}\mathrm{Rb},F=2\to F'=3),\quad
\delta_\mathrm{GABES}=\nu_s-\nu_p+3035.732439\;\mathrm{MHz}.
\]

OPD 단위 GHz, TPD 단위 MHz. F′=2 기준 OPD를 F′=3 기준으로 바꾸면 361.58 MHz 추가 차감. 원문이 Doppler peak 또는 미지정 F′ 기준이면 임의 이동 금지. 원문 δ의 부호가 불명확하면 부호 환산 보류. 주파수 차의 반올림 표기를 MHz 정밀 운전점으로 취급하지 않음.

빔 크기는 `1/e² intensity radius`, diameter, D4σ, FWHM 등을 구분. Gaussian 가정이 명시되지 않은 D4σ나 정의 없는 diameter를 무조건 1/e² 반경으로 변환하지 않음. 기본값으로 채운 빔 크기는 관측값이 아님.

Gain은 입력/출력 기준면과 분모를 보존. Probe gain, conjugate gain, intrinsic gain, effective gain, detector-reference ratio, 이론 fit parameter는 구별. `1/(2G−1)`에서 역산한 gain을 실측 gain으로 만들지 않음. 측정 IDS, 전자잡음 차감 IDS, 검출손실 보정 IDS, power-slope IDS도 별도 수량.

서로 다른 온도·pump power·seed power·스캔의 최대값을 하나의 가상 운전점으로 합치지 않음. 스캔 범위를 Cartesian 데이터로 늘리지 않음. 곡선이 그림에만 있으면 페이지·그림을 연결하고, 숫자로 추출하지 않은 점은 만들지 않음. `literature_summary`는 다른 논문 인용이며 독립 측정 표본에서 제외.

## ⟦AI⟧ 후속 보정·예측 연구에서 사용

전체 자료는 이번 작업에서 **열람한 문헌 풀**. 지금 test라는 이름을 붙여도 untouched external test가 되지 않음. 후속 작업에서 논문/장치 단위의 leave-one-group-out 검증을 설계하고, 가능하면 별도로 고정한 미열람·새 측정 자료에서 외부 예측 확인.

`apparatus_group`은 같은 장치·연구실의 중복 가능성을 알리는 잠정 그룹. 인용한 이전 결과를 독립 실험으로 다시 세지 않도록 확인. 사전 인과관계·공동 장치 여부는 원문 근거에 따라 후속 검토 가능.

생성기의 `has_numeric_gain`은 숫자 gain 존재 여부, `canonical_inputs_complete`는 9개 주요 입력 필드 존재 여부. 두 지표 모두 **학습 적합성 인증 아님**. 기준면·미보고 nuisance parameter·측정 불확도·구조 차이 검토 후 사용. 현재 `fit_accepted`는 전부 false. 불완전 자료도 선언한 근사·조건부 추정·민감도 분석에 활용 가능하나, 추정 입력은 보고값과 별도 관리.

요청한 DEVLOG 전략 중 이번 단계에 적용한 부분: 완료 기준 먼저 기록, 기존 원문 재사용, 정보가 필요한 지점만 추가 조사, 단위/정의가 다른 값 분리, 구조 검증 후 독립 수치 검수. 실제 속도 최적화·χ 보정 모델 선택은 자료 구축 이후.

경험 보정은 실험용 FWM의 허용 도구. 이 DB나 이후 fitted χ의 성공을 GC의 no-fit/미시 완결성 달성으로 자동 승격하지 않음. GC에는 미지 항의 형태·민감도·반례를 제안하는 탐색 근거로 사용 가능.

## ⟦AI⟧ 재생성·검증

저장소 루트에서:

```powershell
python references/fwm_experiments/build_db.py
python references/fwm_experiments/build_db.py --check
python references/fwm_experiments/verify_db.py
```

검사: ID 중복, 단위/수치 유형, 구간 경계, 수치 출처·원문 위치, 환산식, PDF 형식·SHA-256, 기존 10편 보존. 원문 의미의 정확성은 별도 [수동 검수 기록](REVIEW.md)에서 확인. 생성 결과가 FAIL이면 사용 전 수정. `_build_new_records.py`는 추가7편 최초 수작업 큐레이션의 재현 기록. 일상 갱신은 canonical JSON을 편집한 뒤 `build_db.py`만 실행.

검증 범위는 새 DB와 열람본. GABES 계산 코드·UI·의존성 변경 없음. 수치 엔진·GC 회귀 캠페인은 이 자료 작업의 검증 범위에 포함하지 않음.
