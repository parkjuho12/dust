# PM10 실시간 예측 및 온톨로지 기반 행동 조언 시스템

서울시 25개 자치구를 대상으로 1시간 뒤 PM10(미세먼지) 등급을 예측하고, 사용자 맥락에 따른 맞춤형 행동 조언을 제공하는 시스템입니다.

## 시스템 구성

```
사용자 입력
    ↓
[n8n] 온톨로지 기반 슬롯 수집 (user_role / activity_type / sensitive_group / region)
    ↓
[FastAPI] 실시간 PM10 등급 예측 (기상청 + 에어코리아 API → XGBoost)
    ↓
[n8n] REG-LEG 추론 프레임워크 → 맞춤형 행동 조언 출력
```

## 주요 특징

- **ML 예측**: XGBoost 기반 4등급 분류 (좋음 / 보통 / 나쁨 / 매우나쁨)
- **실시간 서빙**: FastAPI + 기상청/에어코리아 API 실시간 연동
- **설명 가능한 추론**: 온톨로지 슬롯 + 환경부 규정 기반 결정론적 REG-LEG 프레임워크
- **대화형 인터페이스**: n8n 워크플로우 기반 멀티턴 대화

## 데이터 출처

| 데이터 | 출처 | 기간 |
|--------|------|------|
| PM10/PM2.5/NO2 측정값 | 에어코리아 | 2023~2024 |
| 기상 관측 데이터 | 기상청 (서울 108번 지점) | 2023~2024 |

## 프로젝트 구조

```
├── train_23_24_final.ipynb   # ML 학습 파이프라인
├── main.py                   # FastAPI 서비스
├── models/
│   ├── tree/                 # XGBoost, RandomForest 모델
│   ├── tensorflow/           # TensorFlow 모델
│   └── preprocess/           # 스케일러, 피처 목록
├── outputs/
│   └── plots/                # 시각화 결과
└── docs/
    ├── ml_pipeline.md        # ML 파이프라인 상세
    └── api_service.md        # FastAPI 서비스 상세
```

## 상세 문서

- [ML 파이프라인](docs/ml_pipeline.md)
- [FastAPI 서비스](docs/api_service.md)
