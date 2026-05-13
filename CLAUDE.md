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

## Ablation Study

모델 각 구성 요소의 기여도를 검증하기 위해 아래 세 가지 축으로 ablation study를 수행한다.

| 구분 | 실험 변인 A | 실험 변인 B |
|------|------------|------------|
| 다양성 표현 방식 | Feature space diversity (microLens-100k의 modality feature 기반) | Categorical diversity (장르 등 범주형 속성 기반) |
| Alpha 추정 방식 | Context-aware alpha (사용자 맥락을 입력으로 동적 예측) | Static alpha (고정된 단일 값 사용) |
| 보상 구성 | Immediate + retention reward | Immediate reward only |
