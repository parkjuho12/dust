# FastAPI 서비스

## 1. 개요

학습된 XGBoost 모델을 기반으로 1시간 뒤 PM10 등급을 실시간으로 예측하는 REST API 서비스입니다. 기상청과 에어코리아 API를 호출하여 실시간 데이터를 수집하고 피처를 구성합니다.

---

## 2. 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

환경변수 설정 필요 (`.env`):
```
KMA_API_KEY=기상청_API_키
AIR_API_KEY=에어코리아_API_키
```

---

## 3. 엔드포인트

### `GET /`
서비스 상태 및 모델 준비 여부 확인

```json
{
  "message": "✅ 1시간 뒤 PM10 등급 예측 API",
  "model_ready": true,
  "feature_count": 153,
  "supported_regions": ["서울 강남구", "서울 관악구", ...],
  "env_ready": { "KMA_API_KEY": true, "AIR_API_KEY": true }
}
```

### `GET /predict?region=서울 중구`
1시간 뒤 PM10 등급 예측

**요청 파라미터**

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| region | string | 서울 중구 | 서울 자치구명 |

**응답 예시**

```json
{
  "status": "success",
  "prediction": "보통",
  "prediction_code": 1,
  "pm_level_code": "NORMAL",
  "prediction_proba": {
    "좋음": 0.12,
    "보통": 0.71,
    "나쁨": 0.14,
    "매우나쁨": 0.03
  },
  "warning": false,
  "warning_message": null,
  "current_pm10": 45.0,
  "station": "중구",
  "station_code": 111121,
  "network_type": "도시대기",
  "requested_region": "서울 중구",
  "data_time": "2024-04-18 13:00"
}
```

---

## 4. 데이터 수집 흐름

```
GET /predict?region=서울 중구
        ↓
지역 → 측정소 매핑 (REGION_TO_STATION)
        ↓
기상청 API 호출 (현재 / 1시간 전 / 2시간 전)
        ↓
에어코리아 API 호출 (해당 측정소 실시간)
        ↓
피처 구성 (lag, 강수, 시간 파생변수 등)
        ↓
StandardScaler 변환
        ↓
XGBoost 예측 (threshold=0.30 적용)
        ↓
JSON 응답 반환
```

---

## 5. 지역 매핑

서울 25개 자치구를 에어코리아 도시대기 측정소와 매핑합니다.

| 지역 | 측정소명 | 측정소코드 |
|------|----------|-----------|
| 서울 중구 | 중구 | 111121 |
| 서울 종로구 | 종로구 | 111101 |
| 서울 용산구 | 용산구 | 111111 |
| 서울 강남구 | 강남구 | 111261 |
| 서울 관악구 | 관악구 | 111311 |
| ... | ... | ... |

지역명은 `"서울 중구"` 또는 `"중구"` 두 형식 모두 허용됩니다. 매핑 실패 시 서울 중구로 대체하고 경고 로그를 남깁니다.

---

## 6. 예측 로직

### 등급 분류 기준

| 코드 | 등급 | PM10 (㎍/m³) |
|------|------|-------------|
| 0 | 좋음 (GOOD) | ≤ 30 |
| 1 | 보통 (NORMAL) | 31~80 |
| 2 | 나쁨 (BAD) | 81~150 |
| 3 | 매우나쁨 (VERY_BAD) | > 150 |

### Threshold 적용
매우나쁨 예측 확률이 **0.30 이상**이면 argmax 결과와 무관하게 매우나쁨으로 강제 분류합니다. 이는 매우나쁨 클래스의 낮은 사전 확률로 인한 과소 예측을 보정합니다.

```python
if probs[3] >= 0.30:
    pred_code = 3
else:
    pred_code = int(probs.argmax())
```

---

## 7. 피처 구성

추론 시 학습과 동일한 피처 구조를 재현합니다.

| 피처 그룹 | 내용 |
|-----------|------|
| PM10 시차 | pm10_lag1/2/3 (에어코리아 과거 측정값) |
| 대기질 | PM2.5, NO2 |
| 기상 | 습도, 지중온도(20cm/30cm), 현상번호, 적설 |
| 강수 파생 | rain_binary, precip_class, precip_weighted, precip_lag1/2 |
| 시간 | hour, month, dayofweek, season |
| 측정소 | 측정소코드, 망 (One-Hot) |

학습 시 저장된 `dust_selected_features_final.pkl` 기준으로 `reindex`하여 컬럼 구조를 맞춥니다. 없는 컬럼은 0으로 채웁니다.

---

## 8. 에러 처리

| 상황 | HTTP 코드 | 내용 |
|------|-----------|------|
| 모델 미로드 | 500 | 모델 파일 없거나 로드 실패 |
| API 키 없음 | 500 | 환경변수 미설정 |
| 기상청 API 실패 | 503 | 타임아웃 또는 응답 오류 |
| 에어코리아 API 실패 | 503 | 타임아웃 또는 응답 오류 |
| region 비어있음 | 400 | 빈 문자열 입력 |
