# ⟦AI⟧ 단일 빔 비선형 전파

사용자 채택 범위: ⁸⁷Rb D1 단일 빔에서 blue/red self-(de)focusing 검증 그림 생성.
2026-11-16에는 물리·수치 검증 후 SABES UI를 연결하기로 결정. 2026-11-17 사용자 요청으로 선택형 UI 연결 완료. 아래 세부 조건·수치 기준은 구현상의 선택이며 실험 조건으로 채택된 값 아님.

## SABES UI 사용

2026-11-17 사용자 요청으로 검증된 단일 빔 경로를 선택형 profiler에 연결.

1. SABES에서 **Instrument → Beam profiler → Rb single-beam** 선택. 캔버스 측정점 선택 없이 사용 가능. 기존 **Gaussian optics**는 기존 기하 광학 판독 유지.
2. **Input beam** 선택. **Custom single beam**은 D1/D2·주파수 기준·디튜닝·혼합비·온도·셀 길이·입구 광출력/반경/곡률을 직접 입력. 처음에는 ⁸⁷Rb D1 blue 기준 조건 적용; **Load red reference / Load blue reference**로 기준 복원.
3. **Cell & laser**, **Numerical checks**에서 조건 확인 후 **Run single-beam calculation** 실행. 물리 입력이 바뀌면 이전 결과를 숨기고 재실행 요구. 자동으로 무거운 원자 계산을 시작하지 않음.
4. 실행 후 **Camera distance after cell (mm)**와 **Profile view** 변경. 카메라 단면, 셀부터 카메라까지 반경, 2D 세기 표시. 카메라 위치·표시 방식 변경은 저장된 셀 출구장을 재사용하며 원자 응답 재계산 없음.

**Cell pump / Cell seed line**은 셀 입사 전 source chain의 한 광주파수 성분만 가져옴. Seed는 **Isolated seed component**로 선택하며, 선택한 선의 주파수·광출력 사용. 전체 스펙트럼 광출력을 한 선에 합치지 않음. 다른 성분·다른 arm의 제외 광출력은 메타데이터에 기록. 혼합비와 reset은 독립 profiler 설정이며, 이 선택으로 실제 다중 빔 FWM 계산을 수행하는 것은 아님.

Source chain의 Gaussian은 명목상 **셀 중심면** 기준. −L/2만큼 역전파하여 셀 **입구면 반경·파면 곡률**로 변환. 기존 기하 계산 파장은 출처에 보존하고, 새 장 전파에는 선택 성분의 실제 광주파수에 대응하는 파장 사용. 실제 waist 위치가 측정됐다는 뜻은 아님. M²=1·선편광만 허용; 타원편광·M²≠1·생성된 probe/conjugate를 입사장으로 재사용하는 경로는 거절. 사용자 입력의 양의 곡률 R은 발산 파면.

### 계산 범위와 판정

- 한 CW 광주파수, 선편광, B=0, 국소 정상상태 full OBE·Maxwell 평균. 모든 혼합비와 선택한 D-line의 초미세/Zeeman 상태 포함. 다른 D-line의 먼 주파수 기여, 공동 포화·Raman/FWM, 열렌즈·원자 횡이동은 미포함.
- 같은 입사장으로 **Full response / Absorption only / Vacuum** 계산. 카메라는 셀 출구 이후 자유공간; 장치의 후단 렌즈·손실·FWM gain을 추가하지 않음.
- 감수율 표는 직접 OBE 중간점과 속도 차수·cutoff 변화 검사. 기본 세기 상한은 입구 peak의 4배; 이를 넘으면 외삽하지 않고 중단. 상한·격자 창·분해능을 조정한 뒤 재실행.
- **Check propagation convergence** 기본 켜짐. 횡격자·z 간격·계산 영역·세기 표를 각각 세분화하여 반경·투과율·복소 출구장·가능한 위상 곡률 비교. 원자 검사, 셀 수렴, 에너지·경계 검사를 구분해 기록. 카메라에서는 자유공간 에너지·경계 검사 추가. 임의 카메라 평면의 반경·프로파일 수렴을 독립적으로 검사한 것은 아니며, 내보내기에 `camera_convergence.assessed=false`로 구분.
- 수렴 옵션을 끄면 **Exploratory**; 검사 실패 시 성공으로 표시하지 않음. 기존 검증 그림의 통과가 새 입력 전체의 통과를 대신하지 않음.
- 출력 반경은 2차 모멘트 반경. 출력이 입사 광출력의 10⁻¹² 미만이면 반경·위상 사용 불가. 이차 위상 적합이 나쁘면 등가 초점거리 사용 불가. 렌즈 부호는 수렴 오차보다 충분히 클 때만 집속/발산으로 표시.

원자 표·셀 전파·카메라 위치는 별도 캐시. 실제 주파수, 혼합비, 온도/reset, 세기 상한, 속도 격자, 셀 길이와 수치 설정을 구분하고 원자·전파 소스 해시 및 NumPy/SciPy 버전으로 무효화. 주파수 기준 선택만 바꾸면 실제 레이저 주파수는 보존됨.

### 내보내기와 구현

**Conditions, convergence & export**에서 조건·판정 확인.

- **Download conditions & checks** → `sabes_single_beam.json`: 주파수 기준, 입구면·제외된 광성분, 물리/수치 조건, 카메라 거리, 검사 결과, 소스 지문.
- **Download field arrays** → `sabes_single_beam.npz`: full/흡수-only/진공의 복소 셀 출구장·카메라장, 좌표, z별 반경·광출력·중심 단면 및 같은 JSON 메타데이터.

공용 계산: `gabes/single_beam_profile.py`. 입사 성분·면 변환: `sabes/single_beam.py`. UI: `gabes_ui/sabes/beam_profiler.py`. 생산 코드가 `gabes_lab`를 import하지 않으며, 기존 파이프라인도 공용 위상 적합·복소 장 비교를 호출.

2026-11-17 변경 범위 검사: **392 passed, 30.23 s**. SABES·새 profiler/AppTest·주파수·비선형 응답·paraxial·관련 파이프라인·저장소 크기 검사 포함. 이번 UI 변경의 전체 저장소/GC suite 통과 주장은 아님.

실제 blue UI 기본 조건 검사: ⁸⁷Rb D1 상부 F=2 중심 +1500 MHz, 100 °C, 50 mm, 5.026548 mW, 입구 반경 0.4 mm·평면 파면, reset 5×10⁵ s⁻¹. 창 반폭 2.4 mm, 128² 격자, z 80단계와 네 가지 세분화 통과. 셀 뒤 500 mm 반경 **0.431914404 mm**, 상대 렌즈 세기 **+2.98024 m⁻¹**. 원자 표 39.82 s, 셀·세분화 4.77 s, 카메라 0.099 s는 해당 환경의 측정값. 기존 benchmark 반경과 0.00402% 차이; 실제 laser k와 별도 수치 격자 사용.

검사 기록: `runs/single_beam_ui_validation/20261117T031658Z-b992d53/pytest.log`.
계산 기록: `runs/single_beam_profile_ui_smoke/20261117T030905Z-b992d53/smoke.json`.
실험 검증 또는 모든 연속 입력에 대한 보증 아님.

## 기존 ⁸⁷Rb D1 검증 파이프라인

저장소 루트:

```text
python -m gabes_lab.pipelines.beam_self_action
```

출력은 `runs/beam_self_action/<run>/`. `--out`으로 지정 가능.
결과 파일: `self_action_validation.png/pdf`, `propagation.npz`,
`susceptibility_tables.npz`, `convergence.json`, `source_manifest.json`, `run.json`.
`--skip-refinement`은 탐색용이며 검증 통과로 표시하지 않음.

## 모델과 부호

- 원자 응답: 기존 `psr_full_obe.solve_psr_pump`의 16상태 full OBE.
  `nonlinear_response.solve_rb87_d1_mean`이 공용 속도 격자로 Maxwell 평균.
  RF·Langevin 계산은 수행하지 않음.
- 순수 ⁸⁷Rb D1, x 선편광, B=0, 정상상태 CW, 한 광주파수.
- 디튜닝: 레이저 주파수 − 상부 ground F=2 전이 세기 가중 중심.
  양수 blue. 실제 F=2→F′ 성분은 이 중심 양쪽에 있으므로 중심은 단일 전이 주파수가 아님.
- 전파장 A의 |A|²는 W/m². 광학 carrier는 exp(ikz−iωt).
  ∂A/∂z = i∇⊥²A/(2k) + ikχ(I)A/2. Imχ>0이면 감쇠.
- `paraxial`은 FFT Strang 분할과 세기 변화를 반영한 exponential midpoint 사용.
  가장자리 흡수막·재정규화 없음. 주기 경계에 빛이 도달하면 오류.
- 감수율 표는 실수부·허수부 각각 PCHIP. 음수 세기·범위 밖 외삽 거절.
- 출력 반경은 광축 기준 sqrt(2⟨x²+y²⟩). 중심 Gaussian에서 1/e² 반경과 동일.
  비Gaussian 출력에는 Gaussian waist라고 부르지 않음.
- 비선형 렌즈 세기는 셀 출구 full/흡수-only 상대 위상을 중심 구간에서 이차식 적합.
  양의 1/f가 집속. 적합 범위와 위상 잔차를 함께 기록.

## 재현용 기본 조건

100 °C, 셀 길이 50 mm, 입구 waist 반경 0.4 mm, 평면 파면, M²=1.
입구 peak intensity 2 W/cm² → 광출력 약 5.027 mW.
디튜닝 ±1.5 GHz. 고정 thermal reset 5×10⁵ s⁻¹.

온도는 밀도와 Doppler 폭에 공통 적용. reset은 선택한 유효 교체율이며 실측값 아님.
집속에 따라 reset을 바꾸지 않음. 원자 횡방향 이동·비국소 기억, 열렌즈,
buffer gas, 충돌 넓어짐·spin exchange, 창 광학과 결합 FWM장은 포함하지 않음.
원자 포화·초미세/Zeeman 광펌핑을 포함한 국소 모델의 수치 검증이며 장치 실험 검증 아님.

## 검증 구조

- 해석해: 진공 Gaussian의 복소 진폭·폭, 균일 복소 slab의 감쇠·위상,
  얇은 렌즈 부호, 포화흡수의 암시적 해와 2차 수렴.
- 원자 응답: 기존 OBE 일치, 독립 약한 빛 complex Voigt, 감수율 표 중간점의 직접 OBE,
  속도 차수·Maxwell cutoff 독립 변화.
- 실제 전파: full, 세기 의존 흡수만 남긴 대조군, 선형 매질, 진공 비교.
- 횡격자 밀도·계산 영역·종방향 간격·세기 표를 각각 세분화.
  빔 폭·투과율·프로파일과 비선형 위상 곡률 확인.
- 상세 기준·통과 여부는 각 실행의 `convergence.json` 기준.
  미통과 시 그림을 성공 결과로 승격하지 않음.

검사:

```text
python -m pytest -q tests/test_paraxial.py tests/test_nonlinear_response.py tests/test_psr_full_obe.py
```

## 참고

- [McCormick et al. (2003)](https://arxiv.org/pdf/physics/0309017):
  Rb D2 blue 집속/red 발산 및 흡수·굴절 분리 실험. 본 D1 실행의 보정 데이터 아님.
- [Kardum et al. (2026)](https://www.mdpi.com/2304-6732/13/7/672):
  복소 굴절률 표와 split-step 전파. 본 계산은 GABES 원자 엔진 사용.

## ⟦AI⟧ 모든 Rb 혼합비·D1/D2 확장

사용자 추가 채택 범위: ① 공통 full OBE, ② 네 isotope×D-line 검증, ③ 연속 혼합비와 주파수 기준 통일, ④ 전체 초미세 구조·전파 수렴 검증. 기존 ⁸⁷Rb D1 그림과 API 유지. 당시 보류한 SABES UI 연결은 2026-11-17 별도 요청으로 위 선택형 단일 빔 모드에 구현.

```python
from gabes.optical_frequency import RbOpticalFrequency
from gabes.nonlinear_response import solve_rb_mean

frequency = RbOpticalFrequency.from_detuning(
    "Rb85", "D2", -250e6, reference="hyperfine_transition",
    ground_f=3, excited_f=4,
)
response = solve_rb_mean(
    frequency, 2e4, rb85_fraction=0.7217,
    temp_c=70., ground_relaxation_hz=5e5,
)
chi = response["chi"]
```

- `rb85_fraction`: 0~1 연속 입력. 0은 순수 ⁸⁷Rb, 1은 순수 ⁸⁵Rb. 총 원소 밀도 N을 fN과 (1−f)N으로 한 번 분배.
- 두 동위원소에 동일한 실제 광주파수·국소 세기 적용. χ와 편극을 합산한 뒤 하나의 빔 전파. 각각 전파한 폭·투과율 평균은 사용하지 않음.
- 각 동위원소의 모든 ground/excited F,m과 광펌핑·포화 포함. 상태 수: ⁸⁷Rb D1 16, ⁸⁵Rb D1 24, ⁸⁷Rb D2 24, ⁸⁵Rb D2 36.
- `optical_bloch.solve_optical_pump`: 공통 full OBE. 새 경로는 전체 Liouvillian의 정확한 희소 LU; 준위 삭제·약한 여기 근사 없음. 기존 PSR wrapper는 원래 dense 연산 순서 유지.
- 주파수 객체는 고정 ⁸⁵Rb fine centroid + offset으로 저장. 혼합비 변경 시 축 이동 없음. 기준은 fine centroid, 상·하부 ground-F 세기 중심, 개별 허용 초미세 전이 중 선택. Hz↔MHz·rad/s 경계 명시.
- 새 경로는 실제 레이저 k를 Doppler·흡수·전파에 공통 적용. 기존 `solve_rb87_d1_mean`은 기존 isotope-centroid k 유지. 따라서 두 경로의 극소한 Doppler 차이는 의도된 주파수 통일 효과.
- SABES carrier 메타데이터는 FWM OPD 정의인 ⁸⁵Rb D1 F=2→F′=3 기준으로 수정: 기존 값에서 +1921.50267275 MHz. 기존 Gaussian 광학 계산 파장과 FWM OPD 수치는 유지. `SourceChain.frequency_metadata`에 실제 파장과 기존 기하 계산 파장 구분.

실행:

```text
python -m gabes_lab.pipelines.rb_spectrum_validation
python -m gabes_lab.pipelines.rb_beam_validation
python -m gabes_lab.pipelines.rb_beam_validation --lines D2 --offsets-mhz -2500 1000 --fractions 0 0.3 0.7217 1
```

`--offsets-mhz`는 해당 D-line의 고정 ⁸⁵Rb fine centroid 기준. 단일 전이 기준 입력은 Python 주파수 변환 API 사용. 임의 주파수에서 새 원자 응답 계산하며, 고정된 주파수 표의 외삽으로 범위를 늘리지 않음.

스펙트럼 파이프라인은 전체 20개 허용 초미세 전이의 중심·주변·외곽을 검증. 독립 complex Voigt와 약한 빛 OBE, 유한 세기 응답, 속도 차수와 cutoff 비교. 빔 파이프라인은 순수 양끝·중간·자연 혼합비에 대해 세기 표 중간점 직접 OBE, 횡격자·영역·z 간격·세기 표·약한 빛 한계를 검사. 조건·원시 배열·그림·소스 해시·수렴 판정은 각 `runs/` 폴더에 기록.

스펙트럼 전체에 단순 red 발산/blue 집속 부호를 강제하지 않음. 전이 중첩·광펌핑이 포함된 실제 계산 부호 보고. 흡수로 출력이 수치적으로 소실되면 반경·위상은 unavailable; 에너지·경계 검사는 계속 적용. 이차 위상 적합이 나쁘면 등가 초점거리는 보고하지 않고 복소 장의 수렴을 평가.

지원 범위는 선택한 D1 또는 D2 주변의 모든 초미세 구조와 디튜닝 입력. 먼 주파수에서 다른 fine-structure 선의 기여까지 포함하는 광대역 모델은 아님. 연속 혼합 법칙과 대표점 검증은 모든 혼합비를 수치적으로 전수 검사했다는 뜻도 아님. 여러 pump/seed/sideband가 같은 원자에 작용하는 경우 공동 포화·Raman/FWM 응답이 추가로 필요함.
