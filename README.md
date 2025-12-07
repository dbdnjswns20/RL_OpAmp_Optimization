# RL_OpAmp_Optimization
# Deep RL을 이용한 180nm 2-Stage Op-Amp 자동 최적화

**2025-2학기 강화학습개론 프로젝트 (CSE5516-01)**
* **조:** 40조
* **팀원:** 120250652 유원준
* **소속:** 서강대학교 (Sogang University)

---

## 1. 프로젝트 개요 (Overview)
이 프로젝트는 **심층 강화학습(Deep Reinforcement Learning, PPO)**을 사용하여 180nm 공정의 **2-Stage Operational Amplifier (Op-Amp)** 회로 설계를 자동화하는 프레임워크입니다.
기존의 수동 설계 방식이나 단순 최적화 기법이 해결하기 어려운 **Gain, Phase Margin, Bandwidth, Power, Area 간의 복잡한 Trade-off**를 AI 에이전트가 스스로 학습하여 최적의 트랜지스터 사이징(W/L)과 보상 커패시터(Cc) 값을 도출합니다.

### 핵심 접근법 (Key Approach)
* **Algorithm:** PPO (Proximal Policy Optimization) - Stable Baselines3 사용
* **Simulator:** NgSpice (PySpice 인터페이스 활용)
* **Reward Strategy:** Two-Phase Reward System (Spec 만족 전/후 보상 전략 차별화)
* **Environment:** Custom Gymnasium Environment (`CircuitEnv`)

---

## 2. 주요 성과 (Key Results)
본 저장소에 포함된 모델(`model.zip`)의 시뮬레이션 결과입니다. 
목표 사양을 대부분 충족하며, 특히 **DC Gain(77.76dB)**에서 목표치(60dB)를 크게 상회하는 성능을 보였습니다.

| Specification | Target | **AI Result (Current Model)** | Status |
|:---:|:---:|:---:|:---:|
| **DC Gain** | ≥ 60 dB | **77.76 dB** | ✅ Pass (High Performance) |
| **Phase Margin** | ≥ 60 deg | **56.29 deg** (≈60) | ⚠️ Acceptable |
| **UGBW** | ≥ 50 MHz | **52.48 MHz** | ✅ Pass |
| **Power** | ≤ 0.8 mW | **1.03 mW** | ⚠️ Slight Overhead |
| **Active Area** | ≤ 1000 $\mu m^2$ | **791.12 $\mu m^2$** | ✅ Pass |

> **Note:** 학습 과정에서 Power 0.79mW, Area 656um²의 최적점(Best Checkpoint)을 달성했으나, 본 모델은 **Gain 성능을 극대화(77dB)**하는 방향으로 수렴된 버전입니다.

---

## 3. 설치 및 환경 설정 (Installation)

### 필수 요구 사항
* Python 3.10.11
* Windows OS (NgSpice DLL 호환성 이슈)

### 라이브러리 설치
터미널에서 아래 명령어를 실행하여 필요한 패키지를 설치합니다.
```bash
pip install -r requirements.txt

```

### NgSpice 설정 (필수)
본 프로젝트는 ngspice.dll 파일을 사용하여 시뮬레이션을 수행합니다.
* ngspice.dll과 180nm_bulk.lib 파일이 반드시 main.py와 같은 폴더에 위치해야 합니다. (본 저장소에 포함되어 있음)

---

## 4. 실행 방법 (How to Run)
### 방법 1: 학습된 모델 검증 (Inference)
* 이미 학습된 모델 (model.zip)을 불러와서 최적 설계 값을 확인하고 성능을 검증합니다.
```bash
python check_model.py

```

### 방법 2: 처음부터 학습시키기 (Training)
* AI가 처음부터 회로 설계를 학습하는 과정을 실행합니다.
```bash
python main.py

```

### 방법 3: 자동 학습 및 모니터닝 (Auto-Run)
* 학습 과정을 TensorBoard로 실시간 모니터링하고 메모리 누수를 방지하며 장시간 학습을 수행합니다.
* auto_run.bat 파일을 더블 클릭하세요.

---

## 5. 파일 구조 (File Structure)
| 파일명 | 설명 |
| :--- | :--- |
| `main.py` | **메인 코드**: RL 환경 설정, PPO 에이전트 학습, 시뮬레이션 인터페이스 |
| `check_model.py` | **검증 코드**: 학습된 모델을 불러와 최종 성능을 테스트하는 스크립트 |
| `run_loop.py` | **자동화 코드**: 메모리 관리를 위해 학습 스크립트를 반복 실행하는 루프 |
| `auto_run.bat` | **배치 파일**: 텐서보드와 자동 학습 루프를 한 번에 실행하는 윈도우 스크립트 |
| `model.zip` | **학습 모델**: 최종 성능을 달성한 PPO 모델 체크포인트 |
| `180nm_bulk.lib` | **공정 라이브러리**: 180nm CMOS 파라미터 파일 |
| `ngspice.dll` | **시뮬레이터**: NgSpice Shared Library |
| `requirements.txt` | **의존성**: 필요한 Python 패키지 목록 |

## 6. 참고 사항
> **Note:** 학습 로그 데이터(training_log.csv)는 용량 제한으로 인해 업로드하지 않았습니다. 상세 데이터는 보고서(PPT) 내 그래프를 참조해 주세요.
