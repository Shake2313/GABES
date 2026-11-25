# ⟦AI⟧ 원자의 빔 통과시간 조사 및 공통 모듈 체크리스트

2026-11-25 조사. GABES `ef2afe7`의 등록 스킴 6개와 하위 모드, 역사적 `ODScheme`, AutoOD `974a7b3` 및 고정 GABES `75facf4`를 비교했다. 사용자 요청은 문헌 조사, 현행 사용처 확인, 공통 구현 계획이다. 아래 API와 적용 순서는 설계 제안이다. 기준 실험은 [11월 24일 ⁸⁵Rb D₁ OD 파워 스캔](https://app.notion.com/p/4226cba14fee8170b728e3b469800d43)이다.

## 문헌에서 확인한 적용 범위

| 조건 | 확인한 현상과 설계에 쓰는 근거 |
| --- | --- |
| 무완충 열증기, 유한 파워 흡수 | Häupl 등 §2.4 식 (7)–(9)는 유효 원자 교체율을 $\gamma_t=\sqrt{8k_BT/(\pi m)}/D_{\mathrm{FWHM}}$로 둔다. ⁸⁷Rb D₂, 0.6 T에서 지름 3개를 비교했고 열린 전이가 닫힌 전이보다 지름에 민감했다. 이번 ⁸⁵Rb D₁ 측정에는 비교용 모델로 적용한다. [Häupl 등, NJP 2025, 저자 공개 원문](https://arxiv.org/html/2410.19916v1#S2.SS4) · [출판 DOI](https://doi.org/10.1088/1367-2630/adb77c) |
| 단일 빔 체류시간 분포 | Sagle 등은 무완충 Cs D₁에서 빔 지름과 자기장을 바꿔 $\gamma_t=(1.13\pm0.20)v_{\perp,\mathrm{rms}}/D_{\mathrm{FWHM}}$를 얻었다. §4.2의 원형 경계 생존곡선은 지수 교체의 근사 오차를 보여준다. Lindvall–Tittonen은 열린 전이에 대해 3차원 기하와 속도로 구한 상호작용 시간 분포에 흡수를 평균했다. 둘을 단일률과 궤적 해의 독립 검산에 쓴다. [Sagle 등, J. Phys. B 1996, 식 (8)·Fig. 6](https://www.lehigh.edu/~jph7/website/Publications/SagleSaturation.pdf) · [Lindvall–Tittonen, PRA 2009](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.80.032505) |
| 약한 프로브와 펌프–프로브 SAS | 열린 전이에서는 여기 상태 포화보다 낮은 세기에서도 초미세 광펌핑이 흡수를 바꾼다. Himsworth–Freegarde §III.D 식 (37)–(38)은 별도 Gaussian transit 정의와 펌핑 세기 의존 유효율을 사용한다. SAS의 실측 선폭과 배경 검증에 쓰되, 단일 빔 OD에는 그 계수를 직접 이식하지 않는다. [Siddons 등, J. Phys. B 2008](https://arxiv.org/abs/0805.1139) · [Himsworth–Freegarde, PRA 2010](https://phyweb.phys.soton.ac.uk/quantum/documents/pra81_023423.pdf) |
| Λ-EIT와 코팅 셀 | 무코팅·무완충 ⁸⁵Rb D₁ EIT에는 뾰족한 중심선이 관측됐다. 코팅·Ne 완충 셀에서는 원자가 빔 밖에 있는 동안의 바닥상태 결맞음이 측정에 영향을 준다. 결맞음 수명, 원자 교체시간, EIT FWHM은 각각 정의한다. [Đujić 등, Sci. Rep. 2024, Fig. 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11196283/) · [Budker 등, PRA 2005](https://doi.org/10.1103/PhysRevA.71.012903) |
| 완충기체·확산 | 빔 밖으로 확산했다 재진입한 원자가 결맞음을 운반해 Ramsey narrowing을 일으킨다. 벽에 의한 이완과 빔 횡단, 스핀 파괴는 별도 과정이다. 가우시안 수광 가중과 확산이 만드는 잡음 상관도 실험 대상이다. [Xiao 등, PRL 2006](https://doi.org/10.1103/PhysRevLett.96.043601) · [Lucivero 등, PRA 2017](https://doi.org/10.1103/PhysRevA.96.062702) |
| Rydberg-EIT 대역폭 | 고정된 광 Rabi 주파수에서 probe waist를 바꾼 Cs 실험은 대역폭의 약 $1/w$ 의존성을 보였다. 대역폭을 $A/\tau_t$에 맞춘 계수에는 정상상태 형성도 들어간다. GABES의 ⁸⁵Rb 결맞음 감쇠 계수는 자체 검증이 필요하다. [Hu 등, EPJ Quantum Technol. 2023, Fig. 2(b)](https://doi.org/10.1140/epjqt/s40507-023-00209-7) · [Ju 등, ⁸⁵Rb 기준 실험](https://arxiv.org/abs/2606.04354) |
| 냉각·포획 원자 | 열증기 빔 횡단 모델의 적용 범위 밖이다. 실제 상호작용 길이·시간과 결맞음 감쇠를 별도로 입력한다. 예를 들어 냉각 원자 SFWM에서 모형의 $\gamma_{12}$는 장치의 유효 감쇠다. [Han 등, 냉각 원자 SFWM 실험](https://pmc.ncbi.nlm.nih.gov/articles/PMC4361990/) |

### 단위와 기하 기준

- 지수 교체율 $\gamma_t=1/\tau$의 내부 단위는 **s⁻¹**. UI가 $\gamma_t/(2\pi)$를 kHz로 받는다면 $\gamma_t=2\pi\cdot10^3\,x_{\mathrm{kHz}}$. 이미 s⁻¹로 받은 값은 그대로 쓴다. 지수적인 완전 교체가 광학 결맞음을 감쇠시킬 때 추가 HWHM은 $\gamma_t$ rad/s, 추가 FWHM은 $\gamma_t/\pi$ Hz. 관측 스펙트럼 FWHM은 전체 선모양에서 따로 구한다.
- 가우시안 **세기** $I(r)=I_0e^{-2r^2/w^2}$. $w$는 $1/e^2$ 반경. $d_{1/e^2}=2w$, $D_{\mathrm{FWHM}}=w\sqrt{2\ln2}$, $I_0=2P/(\pi w^2)$. 운동 온도 $T_{\mathrm{kin}}$과 동위원소 질량 $m$으로 열속도를 구한다. 증기압을 정하는 저온부 온도는 별도 입력이다.
- Häupl 식 (8)의 수치 관례를 그대로 적용하면 ⁸⁵Rb, 99 °C, $w=530$ µm에서 $D_{\mathrm{FWHM}}\approx624$ µm, $\tau\approx2.05$ µs, $\gamma_t/(2\pi)\approx77.7$ kHz. 논문의 $\sqrt{8k_BT/(\pi m)}$는 3차원 Maxwell 평균 속력과 수치가 같다. 원형 경계를 가정하면 횡속력의 Rayleigh 평균과 평균 현 길이 $\pi D/4$의 비도 같은 값이다. 이 조합을 한 묶음으로 식별한다. $w/\langle v_\perp\rangle$ 같은 다른 기하 관례는 별도로 명시한다.
- 궤적 기준 해는 충돌 없는 단일 빔에서 $I(t)=I_0\exp[-2(b^2+v_\perp^2t^2)/w^2]$를 따른다. 진입 플럭스, 빔 안의 점유 수, 검출 모드 가중을 각각 정의한다. 예를 들어 측면으로 유입하는 모델의 속력 분포에는 벌크 분포 대비 $v_\perp$ 가중이 붙는다. 궤적상의 원자 상태를 풀고 관측량을 평균한다. 느린 원자·중심을 비껴가는 원자·공간 절단 오차를 확인한다.

## 현행 코드 감사

| 호출 경로 | 현재 값과 물리적 역할 | 통합할 때 보존할 경계 |
| --- | --- | --- |
| **OD/SAS** | `transit_khz=100` 기본값. $\gamma_t=2\pi\cdot10^3x$. 광펌핑 manifold는 열적 교체를 쓰고 현재 Exact weak는 광학 각주파수 FWHM $2\gamma_t$를 더한다. 유한 프로브의 `ProjectedProbe`는 교체와 광학폭을 각각 반영한다. 펌프가 켜진 코팅 셀은 별도 바닥상태 인구 재순환과 벽 $T_1$을 쓴다. | Legacy·이전 Exact·Generic·축방향 분할 셀·초미세 참조 모델의 폭/리셋 차이와 모델 ID를 보존한다. 펌프가 0인 한계와 펌프–프로브 겹침 영역을 각각 모델링한다. [sas.py](../gabes/schemes/sas.py#L543) · [sas_axial.py](../gabes/schemes/sas_axial.py#L122) · [sas_probe_reference.py](../gabes/schemes/sas_probe_reference.py#L88) · [od_model.py](../gabes/od_model.py#L407) · [optical_pumping.py](../gabes/optical_pumping.py#L52) |
| **Λ coherence** · EIT/AT/CPT | `buffer_ground_relax_khz`는 바닥 결맞음만 감쇠한다. $\gamma/(2\pi)$의 kHz를 내부 단위로 변환한다. EIT/AT 기본 $0.01\Gamma/(2\pi)$ (약 60 kHz), CPT는 $0.005\Gamma/(2\pi)$ (약 30 kHz). 결합 빔 직경은 현재 Rabi 스케일 입력이다. | 잔류 결맞음 감쇠를 보존하고, 열증기 기하 항을 추가할 때 계의 결맞음/인구 모형을 명시한다. 광학 군지연은 별도 관측량이다. [absorption.py](../gabes/schemes/absorption.py#L404) · [atoms.py](../gabes/atoms.py#L218) |
| **Rydberg-EIT** | ⁸⁵Rb 유효 증기 온도와 `beam_diameter_mm`로 $0.6v_{\mathrm{mp}}/d$를 계산해 5S–40D와 관련 결맞음에만 더한다. `beam.transit_broadening_mhz`가 $2\pi10^6$으로 나눈 값을 스킴이 같은 인자로 되돌린다. | 0.6 관례와 Rabi 스케일을 보존한다. 전이 선택·정적/Doppler/유한 IF·온도 스윕을 확인하며, 복수 빔의 겹침을 추가할 때 따로 검증한다. [beam.py](../gabes/beam.py#L56) · [rydberg.py](../gabes/schemes/rydberg.py#L625) |
| **Magneto** · Hanle/EIA/NMOR | 등록 경로 `UnifiedMagnetoScheme`의 코팅 셀은 빔 밖으로 $5\times10^4$, 복귀 20, 벽 리셋 3, 빔 내부 추가 리셋 1 s⁻¹를 기본값으로 쓴다. 완충 셀은 결맞음 재진입 없이 수동 리셋 또는 Rb–Ne 원통 확산·스핀 파괴 예산을 사용한다. | 각 영역의 상태와 복귀/벽/순수 결맞음 감쇠를 보존한다. 단일쌍 참조의 kHz 표기·$2\pi$ 변환도 따로 고정한다. [magneto_unified.py](../gabes/schemes/magneto_unified.py#L253) · [magneto_multilevel.py](../gabes/magneto_multilevel.py#L233) · [buffer_relaxation.py](../gabes/buffer_relaxation.py#L41) |
| **FWM Squeezing** | `transit_rate_khz=100`을 변환해 열적 인구 $5/12,\ 7/12$로 교체한다. Raman 충돌 항은 `floor=0`으로 계산한다. 축약 모델·전체 Zeeman 경로에 적용된다. `GAMMA_GG`는 일부 다른 호출에서 역사적 순수 감쇠 기본값이기도 하다. | 스퀴징의 reset은 한 번 적용한다. 평균장과 잡음 모형이 같은 물리 채널을 가정할 때 소스까지 일치시킨다. `fwm_zeeman_response.py`의 기존 작업이 끝난 뒤 그 경로를 공통 모듈에 연결한다. [fwm/common.py](../gabes/schemes/fwm/common.py#L784) · [constants.py](../gabes/constants.py#L51) |
| **FWM Biphoton** | 독립적인 빔 통과 파라미터가 없다. 일부 OD 분산 경로의 경험적 $0.02\Gamma_i$, 광학 군지연, 기준 방출 수명이 각각 있다. | 예측·보정된 시간파형의 물리 정의를 유지한다. 원자 운동을 도입할 경우 신호·아이들러 모드와 상호작용 영역을 별도로 검증한다. [fwm/biphoton.py](../gabes/schemes/fwm/biphoton.py#L335) |
| **PSR** | 기본 `relaxation_hz`는 등방적 교체 $10^3$ s⁻¹. 현재 입력에 빔 크기가 없고, 고전 기울기·RF 응답·전파 경로가 이 값을 쓴다. | 기하 모델에는 독립 측정한 빔 반경이 필요하다. 기존 잡음·확산 모형의 감쇠와 소스 세기를 임의로 함께 바꾸지 않는다. [schemes/psr.py](../gabes/schemes/psr.py#L114) · [psr_rf.py](../gabes/schemes/psr_rf.py#L29) · [psr_propagated.py](../gabes/schemes/psr_propagated.py#L20) |
| **AutoOD 데스크톱** | GABES `75facf4`의 OD v1을 호출한다. `transit_khz=100` 고정, 사용자 입력인 $1/e^2$ 직경을 반으로 나눈다. 새 GABES 기본 D₂는 별도의 자체 충돌폭 변경이 있는 v2 경로다. | 같은 GABES 공통 모듈을 엔진 핀으로 받는다. D₂ 자체 충돌폭 채택은 별도 물리 결정으로 유지한다. AutoOD에 물리식 사본을 두지 않는다. [현행 AutoOD 어댑터](https://github.com/Shake2313/AutoOD/blob/974a7b35122250e5d9351d70b33ba89b79f15e26/autood/physics.py) · [GABES OD 모델 ID](../gabes/od_model.py#L64) |

등록된 이름과 하위 별칭은 [스킴 레지스트리](../gabes/schemes/__init__.py#L17) 기준이다. 호환용 `ODScheme`의 약한 프로브 흡수에는 별도의 원자 교체 항이 없다. FWM의 양자 궤적·유입 경로는 이미 데이터 구조와 플럭스 검증을 갖고 있다. 명시적 경계 유입을 쓰는 궤적에는 같은 원자 교체를 다시 넣을 수 없게 되어 있다. [역사적 ODScheme](../gabes/schemes/absorption.py#L78) · [양자 유입 적분](../gabes/quantum/inflow.py#L38) · [FWM 궤적 경계 검사](../gabes/fwm_quantum/transport.py#L33)

## 공통 모듈 설계

**`gabes/transit.py` 후보.** 단위 변환, Gaussian 반경 관례, 명시한 기하·운동·질량으로부터 속도와 체류 척도를 계산한다. 선택된 공식과 적용 조건은 모델 ID·버전·출처로 추적한다. 내부는 s⁻¹, m, K, kg를 사용한다. 스킴은 실제 적용 위치를 지정한다. 반환 형태는 필요한 곳에서만 도입한다. 계산 결과에는 `rate_s_inverse`, `residence_time_s`, 모델 ID와 입력 관례를 포함할 수 있다. OD는 열적 교체와 광학 결맞음, Rydberg는 지정한 결맞음, Magneto는 빔 안팎의 교환이다. Λ·Biphoton처럼 해당 채널에 기하 모델이 없는 경로에는 적용 여부를 명시한다.

밀도행렬의 열적 교체는 [`AtomModel.with_reset`](../gabes/atoms.py#L66), [`core.thermal_reset_superoperator`](../gabes/core.py#L151), 잡음 채널은 [`quantum.reservoirs.thermal_reset_channels`](../gabes/quantum/reservoirs.py#L112)를 재사용한다. 같은 과정이 인구·결맞음 식에 미치는 영향을 검사하고, 같은 물리적 손실을 독립 감쇠율로 또 더하는 경우를 막는다. Maxwell 속도 표준편차는 기존 `constants.thermal_speed`를 쓴다. 빔 안팎 왕복 솔버와 완충기체 확산은 현재 모듈이 맡는다. 두 곳 이상에서 쓸 단일 빔 궤적 계산이 검증되면 경로·가중 적분만 `gabes/transit_trajectories.py` 후보로 분리한다. 다중 빔 겹침, 벽 충돌, 잡음 상관, 스킴별 OBE는 해당 모델이 맡는다.

## 구현·검증 체크리스트

### A. 현재 계산을 보존하는 공통화

- [x] 위 표에 맞춰 각 호출 경로의 활성 조건, 값, 단위, 적용 연산자, 기준 위치, 모델 ID를 스냅샷에 기록한다. 모든 스킴의 기본·슬라이더 양 끝·기존 저장 설정, 서브모드와 추가 보기의 원시 배열·표시 판독을 저장한다. `gabes_lab.blocks.snapshot`을 재사용한다.
- [x] `gabes/transit.py`에 실제로 여러 호출자가 필요한 단위·기하 함수만 만든다. 공개 함수 `beam.transit_broadening_mhz`와 기존 설정 키가 같은 값을 재현하도록 연결한다. 파라미터 이름과 캐시 키의 레거시 의미를 보존한다.
- [x] 열적 교체가 같은 경로는 기존 reset 구현으로 연결한다. direct와 jump 경로의 차이, FWM 잡음 소스, Magneto 조건부 밝은 영역 정규화와 열적 리필 극한을 확인한다. 레거시 값과 채널을 변경하는 것은 B 단계에서 별도로 검토한다.
- [x] 단위 왕복, $\gamma=0$, $\rho_\mathrm{th}$ 고정점, trace 보존, 양의 준정부호, 감쇠 한 번 적용을 검사한다. $\gamma_t/2\pi=100$ kHz이면 $\tau=1.59155$ µs이며, $\gamma_t=100\,000$ s⁻¹이면 $\tau=10$ µs인 양쪽 계약을 테스트한다. 저장된 모델의 원시 값은 비트 동일을 목표로 한다. 허용된 예외는 상대 차이 $\le 10^{-12}$와 모든 표시 판독 동일이며, 차이는 커밋·체크리스트에 남긴다.
- [x] 검증은 OD/SAS·피팅·축 방향·코팅, Λ의 EIT/AT/CPT, Rydberg linewidth/IF, Magneto bright/dark, FWM tiers·이득/잡음·Biphoton 파형, PSR 고전/RF/전파에 맞춰 진행한다. 필요 시 `tests/test_beam.py`, `tests/test_od_model.py`, `tests/test_sas_od_model.py`, `tests/test_lambda_joint_doppler.py`, `tests/test_rydberg_model_physics.py`, `tests/test_magneto_multilevel.py`, `tests/test_fwm_validation.py`, `tests/test_psr_scheme.py`와 추가 관련 테스트를 고른다. 공통 엔진이 여러 스킴에 연결된 최종 상태에서 `python -m pytest -q --durations=20`를 한 번 실행한다. 아카이브가 사용하는 엔진 변경은 영향 경로와 핀을 인계한다.

⟦AI⟧ 구현 결과. 고정 kHz 입력과 이미 s⁻¹인 입력, Rydberg의 최빈속력·직경 관례와 FWM의 부동소수점 곱셈 순서를 보존했다. 기존 열적 리셋, 코팅 셀의 밝은 영역–어두운 영역 교환, 완충기체 확산은 원래 채널을 쓴다. 등록 스킴과 참조 경로의 스냅샷 304개에서 물리 배열과 과학 판독은 동일했다. 실행시간과 기존 지원범위 경고는 별도로 대조했다.

### B. 물리 변경은 OD/SAS·AutoOD부터 검증

- [x] 11월 24일 자료의 파워 환산, 모드 호핑 마스크, 주파수·바닥 보정을 고정해 현행 모델을 재생한다. 기준은 공통 온도 100.92 °C, 폭 증가 실측 255.7 MHz·계산 203.2 MHz, RMS 4.03%다. 3.18 µs를 쓰고 온도를 104.2 °C로 다시 맞추면 증가량 234.4 MHz, RMS 3.75%다. 99 °C 고정에서는 증가량이 맞아도 두 끝의 절대 폭을 각각 약 94 MHz 과대예측한다. 판정 지표에 두 날개와 절대 폭을 모두 넣는다.
- [x] 무코팅·무완충·단일 선형 편광 Rb의 **펌프 없는 OD**에 Gaussian 지름 기반 교체율을 선택형으로 연결한다. 구 모델과 새 모델 ID를 함께 시험한다. 0 µW와 유한 파워 연속성, D₁/D₂와 혼합비, $w^{-1}\sqrt{T_\mathrm{kin}/m}$ 스케일, OD 입력·피팅·표시·CSV 경로 일치를 확인한다. D₂ 자체 충돌폭은 비교 중 같은 값으로 유지한다. Häupl 식 (9)의 개별 전이 계수는 별도 정규화 계약이므로 기하 식을 재현한 뒤 reset 연산자를 독립적으로 비교한다.
- [ ] 가우시안 궤적 기준 해와 유효율 모델을 동일 입력에서 비교한다. 경계 플럭스·점유 정규화, 공간·속도·빔 중심과의 최소 거리 $b$ 격자 수렴, 런타임을 기록한다. 분포형 OD 작업은 기존 `od-sas-finite-probe-zeeman-and-transit-refinement`에 연결한다.
- [ ] 같은 $P/w^2$에서 두 빔 반경 이상을 비교하고, 최소 한 반경 또는 파워 시리즈를 검증용으로 분리한다. 셀 직전 파워·반경·온도·편광을 함께 기록한다. 흡수 바닥·파워 척도·온도·주파수 보간에 대한 민감도와 결합 파라미터 상관을 계산한다. 기본값 채택 전 검증 지표와 허용오차는 반복 실측의 변동을 보고 정한다. 한 개의 반경에서 $T,\ P$ 척도, $w,\ \gamma_t$를 모두 자유롭게 맞춰 교체시간을 결정하지 않는다.
- [ ] 스킴별 도입 검증을 거친다. SAS는 펌프–프로브 중첩과 서브도플러 피크·잠금 기울기, Λ는 두 광장의 중첩과 코히어런스 선형상, Rydberg는 광학폭과 IF 대역폭, Magneto는 영역 교환·벽·확산 극한과 선형상, FWM·PSR은 평균 신호와 잡음 소스의 일치, Biphoton은 시간파형과 상관을 비교한다. 기존 실험 보정은 모델 변경의 검증값으로 그대로 전용하기 전에 보정 조건 의존성을 확인한다.

⟦AI⟧ 2026-11-25 검증 범위. ⁸⁵Rb D₁ 파워 스캔 11개의 기존 주파수 보정·제외 구간을 고정해 9,581개 표본을 재생했다. 한 반경에서의 실측 비교이므로 반경 의존성과 기본값 채택은 미결이다. 궤적 검산은 얇은 입구 단면의 단일 종속도군에서 14개 자기 부준위 인구를 따라 수행했다. 전체 종속도 분포 및 셀 전파와의 비교는 분포형 OD 항목에 남긴다.

### C. AutoOD 연동 및 기본값 결정

- [ ] 검증된 GABES 커밋을 `release.json`에 고정한다. `autood/physics.py`는 사용자 단위만 변환해 공통 엔진을 호출한다. 모델 ID·유효 교체율·운동 온도·빔 반경·적용 모드를 표시와 CSV 메타데이터에서 추적한다. 기존 CSV와 설정의 재현, 피팅 선택 가능 변수와 식별 가능성, 캐시 분리를 확인한다.
- [ ] 새 공통 모듈을 이론 검토 해시 목록에 포함하고 이론 TeX/PDF·검토 매니페스트·시작 캐시를 갱신한다. 핀 변경 후 소스 실행·스펙트럼·피팅·시작·CSV 왕복 테스트를 수행한다. 공유 ZIP 갱신은 AutoOD 배포 정책을 따른다.
- [ ] 기존 기본값 교체, Rydberg의 계수 0.6 변경, Magneto 교환율 재설정과 같은 수치 결정은 새 검증 자료와 사용자 판단을 기록한 뒤 별도 변경으로 처리한다.

문헌 검색 절차 참고. [Kassis 등, Scientific Agent Skills](https://arxiv.org/abs/2609.00065)
