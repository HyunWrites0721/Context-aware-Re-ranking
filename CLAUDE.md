# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 언어 지침

- 모든 답변과 설명은 **한국어**로 작성한다.
- 코드의 변수명, 함수명, 클래스명, 주석은 **영어**로 작성한다.

## 답변 방식

- 단답형으로 끊지 말고, 지금 어떤 상황인지 맥락을 포함하여 답변한다.
  - 예: 명령어만 던지지 않고, 왜 이 명령어를 실행하는지, 현재 어떤 단계에 있는지 함께 설명한다.
- 단, 불필요한 반복이나 이미 아는 내용을 장황하게 재설명하지는 않는다.

## 연구 개요

Context-aware 추천 시스템을 통해 사용자 retention을 최적화하는 연구다. 사용자의 현재 맥락(context)을 반영하여 콘텐츠를 추천하고, 단순 클릭률이 아닌 장기적 잔존율(retention)을 보상 신호로 학습한다.

## 실험 데이터셋

- **microLens-100k**: 소규모 멀티미디어 추천 데이터셋. 2024년에 추가된 `extracted_modality_features`를 포함한 풍부한 멀티모달 feature를 보유하므로, 다양성은 **feature space diversity**로 정의한다. 아이템 간 유사도를 feature 벡터 공간에서 계산하여 pDPP의 kernel로 사용한다.
- **KuaiRec**: 쾌수(Kuaishou) 플랫폼의 실제 사용자 행동 로그 데이터셋

## 모델 구조

전체 파이프라인은 두 단계로 구성된다.

### 1단계: Candidate Retrieval (SASRec)
- SASRec(Self-Attentive Sequential Recommendation)을 베이스라인으로 사용
- 사용자 행동 시퀀스를 인코딩하여 **top-100 후보 아이템**을 추출

### 2단계: RL-based Re-ranking (pDPP Agent)
- 강화학습(RL) 기반 agent가 top-100 후보를 re-ranking하여 최종 추천 1개를 출력
- Agent는 사용자의 **diversity preference(alpha)**를 예측
- 예측된 alpha를 활용해 **probabilistic Determinantal Point Process(pDPP)** 방식으로 re-ranking 수행

## 학습 환경 및 보상 설계

- **시뮬레이터**: KuaiSim — 사용자 행동을 시뮬레이션하여 RL 학습 환경 제공
- **보상 신호**:
  - `immediate_reward`: 추천 직후 사용자의 즉각적 반응 (클릭, 시청 등)
  - `retention_reward`: 장기적 사용자 잔존율 기반 보상
- 두 보상을 결합하여 단기 만족과 장기 retention을 동시에 최적화

## 프로젝트 파일 구조

아래 구조는 항상 최신 상태로 유지한다. **파일이나 디렉토리를 새로 생성하거나 삭제할 때마다 반드시 이 섹션을 업데이트한다.**

```
LTV/
├── CLAUDE.md                        # 프로젝트 지침 (이 파일)
├── microlens_paper.pdf              # MicroLens 원본 논문 (벤치마크 수치 참조용)
├── requirements.txt                 # Python 의존성
├── train_sasrec.py                  # SASRec 학습 진입점
├── recbole_SASRec.ipynb             # SASRec 실험용 Jupyter 노트북
├── plot_learning_curves.py          # 학습 곡선 시각화 스크립트
├── learning_curves.png              # 학습 곡선 출력 이미지
├── baseline_param_settings.png      # 논문 하이퍼파라미터 설정 스크린샷
├── baseline_performance.png         # 논문 baseline 성능 스크린샷
├── config/
│   └── sasrec_microlens.yaml        # SASRec 학습 설정 파일
├── dataset/
│   └── microlens100k/
│       └── microlens100k.inter      # RecBole 포맷 interaction 데이터
├── data/
│   └── MicroLens-100k_pairs.csv     # 원본 user-item pair 데이터
├── checkpoints/
│   ├── SASRec-May-13-2026_10-28-13.pth   # 1차 학습 체크포인트
│   ├── SASRec-May-14-2026_03-10-45.pth   # 2차 학습 체크포인트
│   └── SASRec-May-15-2026_06-53-10.pth   # 3차 학습 체크포인트 (최종)
├── figures/                         # 실험별 학습 곡선 PNG (로그 파일명 기반)
│   ├── SASRec-microlens100k-May-14-2026_03-10-29-dffa9c.png
│   └── SASRec-microlens100k-May-15-2026_06-52-54-c18cd4.png
├── log/SASRec/
│   ├── SASRec-microlens100k-May-13-2026_10-27-57-87557e.log   # 1차 학습 로그
│   ├── SASRec-microlens100k-May-14-2026_03-10-29-dffa9c.log   # 2차 학습 로그
│   └── SASRec-microlens100k-May-15-2026_06-52-54-c18cd4.log   # 3차 학습 로그
├── log_tensorboard/                 # TensorBoard 이벤트 파일
├── preprocess/
│   ├── microlens_to_kuaisim.py      # microLens → KuaiSim 변환 스크립트
│   └── kuaisim_integration_guide.txt
└── kuaisim_setup_status.md          # KuaiSim 세팅 진행 상황 메모
```

## 논문 벤치마크 수치 (MicroLens-100K, SASRec IDRec)

`microlens_paper.pdf` Table 2 기준:

| 모델 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
|------|-------|---------|-------|---------|
| SASRec (IDRec) | 0.0909 | **0.0517** | 0.1278 | 0.0610 |

실험 이력:

| 실험 | 설정 | Best Valid NDCG@10 | Test NDCG@10 | 논문 대비 |
|------|------|-------------------|--------------|---------|
| 2차 (May 14) | Adam→AdamW, LR=1e-5, WD=0.1, inner=256 | 0.0376 | 0.0273 | 52.8% |
| 3차 (May 15) | AdamW, LR=1e-4, WD=0.01, inner=2048 | 0.0423 | 0.0301 | 58.2% |

현재 최고 결과 (3차, inner_size=2048):

| 지표 | Best Valid | Test |
|------|-----------|------|
| HR@10 | 0.0750 | 0.0544 |
| NDCG@10 | 0.0423 | 0.0301 |
| HR@20 | 0.1021 | 0.0779 |
| NDCG@20 | 0.0491 | 0.0361 |

→ Test NDCG@10 기준 논문 대비 약 **58.2%** 수준 (0.0301 / 0.0517)

## Ablation Study

모델 각 구성 요소의 기여도를 검증하기 위해 아래 세 가지 축으로 ablation study를 수행한다.

| 구분 | 실험 변인 A | 실험 변인 B |
|------|------------|------------|
| 다양성 표현 방식 | Feature space diversity (microLens-100k의 modality feature 기반) | Categorical diversity (장르 등 범주형 속성 기반) |
| Alpha 추정 방식 | Context-aware alpha (사용자 맥락을 입력으로 동적 예측) | Static alpha (고정된 단일 값 사용) |
| 보상 구성 | Immediate + retention reward | Immediate reward only |
