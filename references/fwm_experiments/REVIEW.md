# DB 검수 기록

## 직접 요청 범위

새 문헌 DB 구축. 기존 10편 보존·원문 재확인, 적합한 새 논문 추가. 후속 gain 보정 및 예측 검증에 사용할 실험값·정의·출처 저장.

## ⟦AI⟧ 완료 기준과 결과

| 기준 | 결과 |
|---|---|
| 기존10편 추적·원제 확인 | 10/10. 이전 CSV는 원문 증거로 사용하지 않음 |
| 추가 원문 조사 | 7편 추가; 총17편. IDS 실험15, 비교 자료2 |
| 실험 단위 분리 | 61개 운전점·스캔·비교·요약 기록. 인용 요약1개는 새 측정 아님 |
| 수치 유형 구분 | 1,104개 조건/결과 항목: 보고602, 근삿값185, 환산119, 구간26, 상한8, 하한8, 미보고156 |
| 출처 재현성 | 22개 source 참조, 고유 PDF19개. 판본·페이지·SHA-256 기록 |
| 미래 gain 분석용 숫자 | 단일 gain값은10편·17기록, gain 구간 포함11편·18기록. 독립 표본 수와 같지 않음 |
| 원문 교차 검수 | [독립13검사](REVIEW_recent.json) PASS. 분류 오류1건 수정·재검증 |
| 구조·파일·환산 계약 | [validation.json](validation.json) PASS |
| CSV/JSON·잘못된 입력 차단 | [validation_detail.json](validation_detail.json), `verify_db.py` 결과 참조 |
| 열람본 | 실제 브라우저에서 검색·펼치기·수치 표·원문 링크 확인. 별도 설치 없는 로컬 HTML |

1,104항목에는 반복된 장치 조건과 미보고 표시 포함. 1,104개 독립 실측 데이터라는 뜻 아님. 그림의 곡선을 숫자 점으로 전수 추출한 DB도 아님.

## ⟦AI⟧ 이전 DB의 주요 정정

| 논문/항목 | 새 기록 |
|---|---|
| McCormick2007 cell19mm | 출판본 약9mm. 초기 arXiv 약8mm와 판본 차이 명시 |
| McCormick2007 seed200µW | 사용 가능한 최대 probe power. 최적 IDS의 실제 입력 seed로 단정하지 않음 |
| McCormick2008 gain9–15 | 측정 effective gain9와 모델로 역산한 intrinsic gain9–15 분리 |
| McCormick2008 −8.8dB | electronics+pump scattering의 system-noise correction. electronics-only 분류 수정 |
| Vogl2012 gain7 | 근거 없는 범위 중간값 대신 원문의 typical4–10 구간 |
| Vogl seed branch | 본문·스펙트럼과 Fig.1의 불일치 보존. 임의 minus-branch δ 환산 보류 |
| Liu bandwidth5.5–16.5MHz | 각각100/700mW 끝점과 연결.400mW optimum에 합치지 않음 |
| Wu 출력129/17µW | 서로 다른 seed 파워 실험. probe/conjugate 동시 출력쌍 아님 |
| de Araujo−4dB | vacuum-seeded quadrature homodyne. 직접 IDS label에서 분리 |
| Allen waists700/390µm | 원문의 D4σ diameters1400/780µm 보존. 추가 Gaussian 가정 없이 1/e² 반경 확정 금지 |
| Jain OPD 미보고 | Fig.1(a) 그림의 약0.8GHz 확인. 정확한 F′ 기준은 불명확 |
| Monsa slope | 보고−8dB와 ratio0.135에서 계산한−8.6967dB 함께 보존 |
| Sim 원제·gain·손실 | 원제 정정. gain≈16/≈15와111/8=13.875 구분. system loss8%를 QE92%로 단정하지 않음 |
| Sim−8.7dB | 측정 noise-power slope ratio0.134에 대응. loss-corrected label 아님 |
| Kim2018−9dB | 청색 off-resonance 문헌 비교군. 적색의 세 공명 측정과 분리, 새 독립 측정으로 중복 계수하지 않음 |

## ⟦AI⟧ root의 추가 원문 검수

- Kim2018 원문pp.3–7: 셀12mm, single-resonant89°C/1W/950·300µm/0.45°, reportedδ+16MHz→GABES−16MHz, F′2 기준−687/−771MHz→F′3 기준−1.04858/−1.13258GHz. 측정gain5.6/3.8 및 세 종류 IDS와 일치.
- Qin2012 pp.1–2: 530/330µm는 명시적1/e² 반경.280mW/15µW 입력에서 출력123/111µW, 비율8.2/7.4. 균형 조정 후−5.7dB는 별도 구성.635kHz slope와850kHz best-spectrum 구분.
- Jasperse2011 pp.8–9: 7mm/130°C/400mW/100µW/630·375µm/0.3° 확인.δ의 숫자는 반올림3040MHz spacing의 산술 환산이며 MHz 정밀 실험값 아님.−4.1dB는 pump background 제거 시 예상값.
- Pooser2009 pp.5–7: 85Rb OPD는 Doppler peak 기준. 단순 출력/입력 gain≈5와 매질모델의 inferred gain 구분 확인. 다른 isotope·D2 값은85Rb D1 label에서 제외.
- Glorieux2010 p.6 Fig.7·p.7 conclusion: −9.2±0.5dB는 전자잡음 보정, −11.0±0.7dB는 전파손실 고려. 온도118°C/pump1200mW/Δ750MHz/δ+6MHz 연결 확인.
- Turnbull2013은 이번 원문의 실험 observable이 양팔 gain map이므로 `scope.measurement=two_arm_gain_map`으로 명시. 제목에 squeezed states가 있어도 새 IDS 측정값으로 분류하지 않음.

## ⟦AI⟧ 후속 사용 범위

데이터 구조·원문 연결 완료. 모든 원문 수치의 무오류, 전 조건 보정 정확도, χ의 식별 가능성 또는 GC 달성 판정은 주장하지 않음. 정밀 조건이 빠진 실험은 조건부 추정·민감도 분석에 사용할 수 있으나, 추정 입력은 원문 보고값과 구분할 것.

우선 다음 단계: gain-bearing 운전점의 기준면·미보고 입력 목록 검토 → 필요한 곡선/원자료 수집 → 논문·장치 그룹 단위 검증 설계. 이번 DB를 보고 선택한 validation은 untouched 외부 테스트와 구분.
