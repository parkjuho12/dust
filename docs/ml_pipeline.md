# ML 파이프라인

## 1. 데이터

### 원본 데이터
- **에어코리아**: 서울 25개 자치구 시간별 대기질 측정값 (PM10, PM2.5, NO2, CO, SO2, O3 등)
- **기상청**: 서울 108번 지점 시간별 기상 관측값 (기온, 습도, 풍속, 강수량, 지중온도 등)
- **학습 기간**: 2023년 (train) / **평가 기간**: 2024년 (test)
- **측정소 범위**: 서울 25개 자치구 40개 측정소 (도시대기 + 도로변대기)

### 데이터 규모

| 구분 | 행 수 | 최종 피처 수 |
|------|-------|-------------|
| Train (2023) | 326,968 | 153 |
| Test (2024) | 329,080 | 153 |

### 클래스 분포 (PM10 등급)

| 등급 | 기준 (㎍/m³) | Train | Test |
|------|-------------|-------|------|
| 0 좋음 | ≤ 30 | 157,211 | 185,845 |
| 1 보통 | 31~80 | 143,506 | 131,997 |
| 2 나쁨 | 81~150 | 22,848 | 9,725 |
| 3 매우나쁨 | > 150 | 3,403 | 1,513 |

매우나쁨 클래스가 전체의 약 1%로 심각한 클래스 불균형이 존재합니다.

![클래스 분포](outputs/plots/plot_class_distribution.png)

---

## 2. 피처 엔지니어링

### 시차(Lag) 피처
과거 PM10/PM2.5 측정값을 시차 변수로 활용하여 시계열 패턴을 반영했습니다.

| 피처 | 설명 |
|------|------|
| pm10_lag1 | 1시간 전 PM10 |
| pm10_lag2 | 2시간 전 PM10 |
| pm10_lag3 | 3시간 전 PM10 |
| pm25_lag1 | 1시간 전 PM2.5 |
| pm25_lag2 | 2시간 전 PM2.5 |
| precip_lag1 | 1시간 전 강수량 |
| precip_lag2 | 2시간 전 강수량 |

### 강수 파생 피처

| 피처 | 설명 |
|------|------|
| rain_binary | 강수 유무 (0/1) |
| precip_class | 강수 강도 분류 (없음/약/중/강) |
| precip_weighted | 강수량 가중치 변환 (15mm 이상이면 2배) |

### 시간 피처

| 피처 | 설명 |
|------|------|
| hour | 시간 (0~23) |
| month | 월 (1~12) |
| dayofweek | 요일 (0~6) |
| season | 계절 (Spring/Summer/Autumn/Winter) |

---

## 3. 결측 처리 및 인코딩

- **수치형 컬럼**: 중앙값(median) 대체
- **범주형 컬럼**: 최빈값(most_frequent) 대체
- **범주형 인코딩**: One-Hot Encoding
- **스케일링**: StandardScaler

### 피처 제외 기준
지역명(`지역`)과 측정망(`망`)은 측정소코드(`측정소코드`)와 중복 정보이므로 학습에서 제외했습니다.

---

## 4. 피처 선택

1차 XGBoost 학습 후 feature importance 기반으로 피처를 선택했습니다.

- **기준**: importance > 0.001
- **선택된 피처 수**: 153개

![Feature Importance Top 20](outputs/plots/plot_xgb_importance_top20.png)

pm10_lag1/2/3이 압도적으로 높은 중요도를 보여 과거 PM10 시계열 패턴이 미래 등급 예측에 가장 핵심적인 역할을 함을 확인했습니다. season_Spring이 상위권에 위치한 것은 봄철 황사 영향을 반영합니다.

---

## 5. 상관관계 분석

![Target 상관관계 Top 20](outputs/plots/plot_target_corr_top20.png)

pm10_lag1/2/3이 가장 높은 상관관계를 보이며, PM2.5 시차 변수도 높게 나타났습니다. 강수 변수는 결측률(87%)로 인해 전체 상관관계는 낮으나 강수 발생 시 PM10 감소에 실질적인 영향을 미칩니다. season_Spring과 season_Autumn이 상위권에 위치한 것은 봄철 황사 및 가을철 대기정체 현상과 관련된 것으로 해석됩니다.

---

## 6. 클래스 불균형 처리

**클래스 가중치 (balanced)**

| 클래스 | 가중치 |
|--------|--------|
| 0 좋음 | 0.52 |
| 1 보통 | 0.57 |
| 2 나쁨 | 3.58 |
| 3 매우나쁨 | 24.02 |

---

## 7. 모델 학습

### TensorFlow (Dense)
```
Dense(128, relu) → BatchNorm → Dropout(0.3)
→ Dense(64, relu) → BatchNorm → Dropout(0.2)
→ Dense(4, softmax)
```

### XGBoost
- n_estimators: 300, max_depth: 6, learning_rate: 0.05
- sample_weight 적용 (클래스 가중치)

### RandomForest
- n_estimators: 200, random_state: 42

---

## 8. Threshold 실험

![Threshold 분석](outputs/plots/plot_threshold_analysis.png)

| Threshold | Accuracy | Macro F1 | 매우나쁨 Precision | 매우나쁨 F1 |
|-----------|----------|----------|--------------------|------------|
| 0.10 | 0.8749 | 0.7388 | 0.4232 | 0.5728 |
| 0.15 | 0.8756 | 0.7515 | 0.4730 | 0.6128 |
| 0.20 | 0.8759 | 0.7581 | 0.5023 | 0.6320 |
| 0.25 | 0.8763 | 0.7639 | 0.5294 | 0.6491 |
| **0.30** | **0.8766** | **0.7707** | **0.5608** | **0.6692** |

Macro F1과 매우나쁨 F1이 모두 가장 높은 **0.30을 최종 threshold로 선택**했습니다.

---

## 9. 최종 모델 성능 비교

![모델 성능 비교](outputs/plots/plot_model_comparison.png)

| 모델 | Accuracy | Macro F1 | 비고 |
|------|----------|----------|------|
| TensorFlow | 0.8442 | 0.6519 | val_loss 지속 증가 (과적합) |
| **XGBoost** | **0.8766** | **0.7707** | threshold 0.30 기준 |
| RandomForest | 0.8729 | 0.7345 | - |

XGBoost가 Macro F1 기준 최고 성능을 달성하여 최종 서비스 모델로 채택했습니다.

![Confusion Matrix](outputs/plots/plot_confusion_matrix.png)

---

## 10. 저장 파일

| 파일 | 설명 |
|------|------|
| `models/tree/dust_xgb_model_final_v2.pkl` | 최종 XGBoost 모델 |
| `models/tree/dust_rf_model_final_v2.pkl` | RandomForest 모델 |
| `models/tensorflow/dust_tf_model_final_v2.keras` | TensorFlow 모델 |
| `models/preprocess/dust_scaler_final_v2.pkl` | StandardScaler |
| `models/preprocess/dust_selected_features_final.pkl` | 선택된 153개 피처 목록 |
