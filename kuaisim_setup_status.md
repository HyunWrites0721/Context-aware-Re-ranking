# KuaiSim 환경설정 진행 현황

## 완료된 작업

### 1. KuaiSim 저장소 클론
- 경로: `/home/data/KuaiSim/`
- 구조: `code/` (학습 코드), `dataset/` (데이터 디렉토리)

---

### 2. 전처리 스크립트 버그 수정 (`preprocess/microlens_to_kuaisim.py`)

| 버그 | 원인 | 수정 내용 |
|------|------|-----------|
| 타임스탬프 파싱 실패 | MicroLens 타임스탬프가 ms 단위인데 `unit='s'`로 파싱 → 연도 52145년 overflow | `unit='ms'`로 변경, `time_ms = df['timestamp']` (곱셈 제거) |
| `pd.qcut` labels 불일치 | 분포 쏠림으로 중복 bin edge가 drop되면 bins 수 < labels 수 | `rank(method='first')`로 rank 변환 후 qcut → 항상 q개 bin 생성 보장 |

---

### 3. MicroLens → KuaiSim 포맷 전처리 실행

출력 경로: `/home/data/KuaiSim/data/MicroLens-KuaiSim/`

| 파일 | 행 수 | 설명 |
|------|-------|------|
| `log_microlens.csv` | 719,405 | 상호작용 로그 (KuaiRand-Pure 스키마) |
| `user_features_microlens.csv` | 100,000 | 사용자 특성 |
| `video_features_microlens.csv` | 19,738 | 아이템 특성 |

포맷 검증 통과 (`is_click`, `is_comment` 전부 1, `is_hate` 전부 0, 컬럼 스키마 일치)

---

### 4. MicroLens 전용 Reader 작성 (`KuaiSim/code/reader/MicroLensSeqReader.py`)

`KRMBSeqReader`를 상속하며 두 가지 문제를 수정:

| 문제 | 원인 | 수정 |
|------|------|------|
| `tag` 타입 오류 | `tag` 컬럼이 int로 저장되어 `.split(',')` 호출 시 `AttributeError` | `get_item_meta_data`에서 `str(tag)` 변환 후 split |
| `get_response_weights` ZeroDivisionError | `is_click`, `is_comment`, `long_view`가 전부 1이라 0의 개수 = 0, `counts[0]` → KeyError | `counts.get(0, 0)`, `counts.get(1, 0)`으로 안전하게 처리 |

---

### 5. UIRM 학습 스크립트 작성 (`KuaiSim/code/train_uirm_microlens.sh`)

- Reader: `MicroLensSeqReader`, Model: `KRMBUserResponse`
- stdout을 `.model.log`로 리다이렉트 → 이후 RL 환경이 이 파일로 reader/model 복원
- 체크포인트 저장 경로: `output/MicroLens/env/model/`

---

## 앞으로 해야 할 작업

### Step 1. UIRM 학습 실행

```bash
cd /home/data/KuaiSim/code
bash train_uirm_microlens.sh
```

이 단계가 완료되어야 KuaiSim 환경이 사용자 반응을 시뮬레이션할 수 있다.  
예상 출력: `output/MicroLens/env/log/*.model.log`, `output/MicroLens/env/model/*.checkpoint`

---

### Step 2. KuaiSim 환경 인터페이스 파악

우리는 KuaiSim의 DDPG/TD3 기반 policy를 사용하지 않는다.  
KuaiSim은 **시뮬레이션 환경**으로만 사용하고, 우리 pDPP Agent가 직접 `env.step()`을 호출한다.

확인이 필요한 항목:
- `env.reset()` 반환 형식 (초기 observation 구조)
- `env.step(action)` 입력 형식 (action = 추천 아이템 목록인지, 인코딩된 ID인지)
- `step()` 반환값: `(observation, reward, done, info)` 중 reward 구조 확인
  - `immediate_reward` 계산 방식
  - `retention_reward` 계산 방식 (`KRUserRetention` 모델 포함 여부)
- candidate item pool 접근 방법 (`env.candidate_ids`, `env.candidate_item_encoding` 등)

---

### Step 3. 커스텀 학습 루프 설계 (`train_pdpp_agent.py`)

KuaiSim env + 우리 pDPP Agent를 연결하는 학습 루프 작성.  
SASRec top-100 후보 → pDPP re-ranking → env.step() → reward 수집 흐름.

```
[SASRec] → top-100 candidates
    ↓
[pDPP Agent]
  - alpha predictor (context → diversity preference)
  - pDPP re-ranking (candidates × alpha → final recommendation)
    ↓
env.step(action) → immediate_reward, retention_reward
    ↓
RL update (alpha predictor 파라미터 갱신)
```

---

### Step 4. pDPP Agent 구현

- **Alpha Predictor**: 사용자 상태(history, context)를 입력받아 diversity preference α를 출력하는 네트워크
- **pDPP Kernel**: SASRec 임베딩 기반 아이템 간 유사도 행렬 (feature space diversity)
- **pDPP Sampling**: α로 parameterize된 확률적 DPP에서 추천 아이템 샘플링

---

### Step 5. Ablation Study 구성

| 실험 | 다양성 표현 | Alpha 방식 | 보상 구성 |
|------|------------|-----------|---------|
| Full model | Feature space (modality) | Context-aware | Immediate + Retention |
| Ablation A | Categorical (tag) | Context-aware | Immediate + Retention |
| Ablation B | Feature space | Static α | Immediate + Retention |
| Ablation C | Feature space | Context-aware | Immediate only |

---

## 파일 위치 정리

| 역할 | 경로 |
|------|------|
| 전처리 스크립트 | `/home/data/LTV/preprocess/microlens_to_kuaisim.py` |
| 전처리 가이드 | `/home/data/LTV/preprocess/kuaisim_integration_guide.txt` |
| 전처리 출력 데이터 | `/home/data/KuaiSim/data/MicroLens-KuaiSim/` |
| KuaiSim 커스텀 Reader | `/home/data/KuaiSim/code/reader/MicroLensSeqReader.py` |
| UIRM 학습 스크립트 | `/home/data/KuaiSim/code/train_uirm_microlens.sh` |
| KuaiSim 저장소 | `/home/data/KuaiSim/` |
| SASRec 체크포인트 | `/home/data/LTV/checkpoints/` |
