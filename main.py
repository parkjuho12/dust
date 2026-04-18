from fastapi import FastAPI, HTTPException, Query
import joblib
import pandas as pd
import requests
import datetime
import os
import logging
from dotenv import load_dotenv
from typing import Any, Dict, Optional

load_dotenv()

app = FastAPI(title="1시간 뒤 실시간 미세먼지(PM10) 등급 예측 서비스")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. 모델 및 전처리 파일 로드
# ---------------------------------------------------------
xgb_model = None
scaler = None
selected_features = None
MODEL_READY = False

MODEL_PATH = "./models/tree/dust_xgb_model_final_v2.pkl"
SCALER_PATH = "./models/preprocess/dust_scaler_final_v2.pkl"
FEATURE_PATH = "./models/preprocess/dust_selected_features_final.pkl"

KMA_API_KEY = os.environ.get("KMA_API_KEY")
AIR_API_KEY = os.environ.get("AIR_API_KEY")

GRADE_MAP = {
    0: "좋음",
    1: "보통",
    2: "나쁨",
    3: "매우나쁨"
}

GRADE_CODE_MAP = {
    0: "GOOD",
    1: "NORMAL",
    2: "BAD",
    3: "VERY_BAD"
}

VERY_BAD_FORCE_THRESHOLD = 0.20


def load_artifacts() -> None:
    global xgb_model, scaler, selected_features, MODEL_READY

    try:
        xgb_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        selected_features = joblib.load(FEATURE_PATH)

        if not isinstance(selected_features, (list, tuple)):
            raise ValueError("selected_features 파일이 list/tuple 형식이 아닙니다.")

        MODEL_READY = True
        logger.info("✅ 모델 로드 완료 (사용 변수: %d개)", len(selected_features))

    except Exception as e:
        MODEL_READY = False
        logger.exception("❌ 모델 로드 실패: %s", e)


load_artifacts()


# ---------------------------------------------------------
# 2. 지역 → 측정소명 / 측정소코드 / 망 매핑
#    (에어코리아 도시대기 측정소 기준)
# ---------------------------------------------------------
REGION_TO_STATION = {
    "서울 강남구": ("강남구",  111261, "도시대기"),
    "서울 강동구": ("강동구",  111274, "도시대기"),
    "서울 강북구": ("강북구",  111291, "도시대기"),
    "서울 강서구": ("강서구",  111301, "도시대기"),
    "서울 관악구": ("관악구",  111311, "도시대기"),
    "서울 광진구": ("광진구",  111131, "도시대기"),
    "서울 구로구": ("구로구",  111321, "도시대기"),
    "서울 금천구": ("금천구",  111331, "도시대기"),
    "서울 노원구": ("노원구",  111151, "도시대기"),
    "서울 도봉구": ("도봉구",  111161, "도시대기"),
    "서울 동대문구": ("동대문구", 111171, "도시대기"),
    "서울 동작구": ("동작구",  111341, "도시대기"),
    "서울 마포구": ("마포구",  111181, "도시대기"),
    "서울 서대문구": ("서대문구", 111191, "도시대기"),
    "서울 서초구": ("서초구",  111351, "도시대기"),
    "서울 성동구": ("성동구",  111201, "도시대기"),
    "서울 성북구": ("성북구",  111211, "도시대기"),
    "서울 송파구": ("송파구",  111361, "도시대기"),
    "서울 양천구": ("양천구",  111371, "도시대기"),
    "서울 영등포구": ("영등포구", 111381, "도시대기"),
    "서울 용산구": ("용산구",  111111, "도시대기"),
    "서울 은평구": ("은평구",  111221, "도시대기"),
    "서울 종로구": ("종로구",  111101, "도시대기"),
    "서울 중구": ("중구",    111121, "도시대기"),
    "서울 중랑구": ("중랑구",  111231, "도시대기"),
}

# 자치구명만으로도 조회 가능하도록 역방향 매핑 추가
DISTRICT_TO_REGION = {v[0]: k for k, v in REGION_TO_STATION.items()}


def get_station_info(region: str) -> Dict[str, Any]:
    """
    region 문자열로 측정소명 / 측정소코드 / 망을 반환.
    '서울 중구' 또는 '중구' 형식 모두 허용.
    매핑 실패 시 서울 중구 기본값 사용.
    """
    # 전체 지역명으로 조회
    if region in REGION_TO_STATION:
        name, code, network = REGION_TO_STATION[region]
        return {"station": name, "측정소코드": code, "망": network}

    # '서울 ' 없이 자치구명만 들어온 경우
    district = region.split()[-1] if region else "중구"
    full_region = DISTRICT_TO_REGION.get(district)
    if full_region:
        name, code, network = REGION_TO_STATION[full_region]
        return {"station": name, "측정소코드": code, "망": network}

    # 매핑 실패 → 기본값(중구) + 경고 로그
    logger.warning("⚠️ 알 수 없는 지역: '%s' → 서울 중구로 대체", region)
    name, code, network = REGION_TO_STATION["서울 중구"]
    return {"station": name, "측정소코드": code, "망": network}


# ---------------------------------------------------------
# 3. 공통 유틸
# ---------------------------------------------------------
def ensure_ready() -> None:
    if not MODEL_READY or xgb_model is None or scaler is None or selected_features is None:
        raise HTTPException(status_code=500, detail="모델 또는 전처리 파일이 준비되지 않았습니다.")
    if not KMA_API_KEY:
        raise HTTPException(status_code=500, detail="KMA_API_KEY 환경변수가 없습니다.")
    if not AIR_API_KEY:
        raise HTTPException(status_code=500, detail="AIR_API_KEY 환경변수가 없습니다.")


def safe_float(v: Any, default: float = 0.0) -> float:
    if v in (None, "", "-", "null", "-9.0"):
        return default
    return float(v)


def season_from_month(month: int) -> str:
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    if month in [9, 10, 11]:
        return "Autumn"
    return "Winter"


# ---------------------------------------------------------
# 4. 데이터 수집 함수
# ---------------------------------------------------------
def get_weather_data(target_dt: datetime.datetime) -> Dict[str, float]:
    tm = target_dt.strftime("%Y%m%d%H%M")
    url = (
        f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"
        f"?tm={tm}&stn=108&help=1&authKey={KMA_API_KEY}"
    )

    try:
        res = requests.get(url, timeout=30)
        res.raise_for_status()

        lines = [
            l.strip()
            for l in res.text.split("\n")
            if l.strip() and not l.startswith("#")
        ]
        if not lines:
            raise ValueError("기상청 응답이 비어 있습니다.")

        parts = lines[0].split()

        return {
            "temp": safe_float(parts[11]),
            "wind": safe_float(parts[3]),
            "humid": safe_float(parts[13]),
            "rain": safe_float(parts[15], 0.0),
            "snow": safe_float(parts[18], 0.0),
            "snow_3h": safe_float(parts[17], 0.0),
            "ground_20": safe_float(parts[22]),
            "ground_30": safe_float(parts[23]),
            "weather_code": safe_float(parts[14]),
        }

    except Exception as e:
        logger.exception("기상청 데이터 수집 실패 (%s): %s", tm, e)
        raise HTTPException(status_code=503, detail=f"기상청 데이터 수집 실패: {str(e)}")


def get_air_data(station_name: str) -> Dict[str, Any]:
    url = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    params = {
        "serviceKey": AIR_API_KEY,
        "returnType": "json",
        "stationName": station_name,
        "dataTerm": "DAILY",
        "ver": "1.3",
    }

    try:
        res = requests.get(url, params=params, timeout=5)
        res.raise_for_status()
        payload = res.json()

        items = payload["response"]["body"]["items"]
        if not items:
            raise ValueError("에어코리아 응답이 비어 있습니다.")

        return {
            "station": station_name,
            "PM10_now": safe_float(items[0].get("pm10Value")),
            "PM25": safe_float(items[0].get("pm25Value")),
            "NO2": safe_float(items[0].get("no2Value")),
            "prev_pm10_1h": safe_float(items[1].get("pm10Value")) if len(items) > 1 else 0.0,
            "prev_pm10_2h": safe_float(items[2].get("pm10Value")) if len(items) > 2 else 0.0,
            "prev_pm10_3h": safe_float(items[3].get("pm10Value")) if len(items) > 3 else 0.0,
            "dataTime": items[0].get("dataTime", ""),
        }

    except Exception as e:
        logger.exception("에어코리아 데이터 수집 실패 (%s): %s", station_name, e)
        raise HTTPException(status_code=503, detail=f"에어코리아 데이터 수집 실패: {str(e)}")


# ---------------------------------------------------------
# 5. feature 생성
# ---------------------------------------------------------
def build_features(user_region: str) -> Dict[str, Any]:
    now = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)

    station_info = get_station_info(user_region)

    w_now = get_weather_data(now)
    w_1h = get_weather_data(now - datetime.timedelta(hours=1))
    w_2h = get_weather_data(now - datetime.timedelta(hours=2))
    air = get_air_data(station_info["station"])

    raw_input = {
        "pm10_lag1": air["prev_pm10_1h"],
        "pm10_lag2": air["prev_pm10_2h"],
        "pm10_lag3": air["prev_pm10_3h"],
        "PM25": air["PM25"],
        "적설(cm)": w_now["snow"],
        "3시간신적설(cm)": w_now["snow_3h"],
        "습도(%)": w_now["humid"],
        "현상번호(국내식)": w_now["weather_code"],
        "precip_lag1": 1 if w_1h and w_1h["rain"] > 0 else 0,
        "precip_lag2": 1 if w_2h and w_2h["rain"] > 0 else 0,
        "20cm 지중온도(°C)": w_now["ground_20"],
        "30cm 지중온도(°C)": w_now["ground_30"],
        "NO2": air["NO2"],

        # 강수 파생변수
        "rain_binary": 1 if w_now["rain"] > 0 else 0,
        "precip_class": 0 if w_now["rain"] == 0 else (1 if w_now["rain"] < 3 else 2),
        "precip_weighted": w_now["rain"] * 2 if w_now["rain"] >= 15 else w_now["rain"],

        # 시간 변수
        "month": now.month,
        "season": season_from_month(now.month),

        # 학습 데이터와 일치하는 측정소코드 / 망
        "측정소코드": station_info["측정소코드"],
        "망": station_info["망"],

        "hour": now.hour,
        "dayofweek": now.weekday()
    }

    df = pd.DataFrame([raw_input])

    categorical_for_dummies = ["rain_binary", "precip_class", "month", "season", "측정소코드", "망", "dayofweek"]

    for col in categorical_for_dummies:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df = pd.get_dummies(df, columns=categorical_for_dummies)

    # bool → float 변환
    df = df.astype(float)

    # 학습 feature 구조 맞춤
    df = df.reindex(columns=selected_features, fill_value=0.0)

    return {
        "df": df,
        "air": air,
        "weather": w_now,
        "raw_input": raw_input,
        "timestamp": now.isoformat(),
        "station_info": station_info,
    }


# ---------------------------------------------------------
# 6. 예측 로직
# ---------------------------------------------------------
def predict_one_hour_ahead(input_df: pd.DataFrame) -> Dict[str, Any]:
    input_scaled = scaler.transform(input_df)

    prediction_proba = None

    if hasattr(xgb_model, "predict_proba"):
        probs = xgb_model.predict_proba(input_scaled)[0]

        if len(probs) != 4:
            raise ValueError("predict_proba 결과 클래스 수가 4가 아닙니다.")

        if probs[3] >= VERY_BAD_FORCE_THRESHOLD:
            pred_code = 3
        else:
            pred_code = int(probs.argmax())

        prediction_proba = {
            "좋음": float(probs[0]),
            "보통": float(probs[1]),
            "나쁨": float(probs[2]),
            "매우나쁨": float(probs[3]),
        }
    else:
        pred = xgb_model.predict(input_scaled)
        pred_code = int(pred[0])

    return {
        "prediction_code": pred_code,
        "prediction": GRADE_MAP[pred_code],
        "pm_level_code": GRADE_CODE_MAP[pred_code],
        "prediction_proba": prediction_proba,
        "warning": pred_code == 3,
        "warning_message": "1시간 뒤 매우나쁨 가능성이 있습니다." if pred_code == 3 else None,
    }


# ---------------------------------------------------------
# 7. API 엔드포인트
# ---------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "✅ 1시간 뒤 PM10 등급 예측 API",
        "model_ready": MODEL_READY,
        "feature_count": len(selected_features) if selected_features else 0,
        "supported_regions": list(REGION_TO_STATION.keys()),
        "env_ready": {
            "KMA_API_KEY": bool(KMA_API_KEY),
            "AIR_API_KEY": bool(AIR_API_KEY),
        },
    }


@app.get("/predict")
def predict(region: str = Query(default="서울 중구")):
    ensure_ready()

    if not region.strip():
        raise HTTPException(status_code=400, detail="region 값이 비어 있습니다.")

    try:
        data = build_features(region)
        pred_result = predict_one_hour_ahead(data["df"])

        return {
            "status": "success",
            "prediction_target": "1시간 뒤 PM10 등급 예측",
            "prediction": pred_result["prediction"],
            "prediction_code": pred_result["prediction_code"],
            "pm_level_code": pred_result["pm_level_code"],
            "prediction_proba": pred_result["prediction_proba"],
            "warning": pred_result["warning"],
            "warning_message": pred_result["warning_message"],
            "current_pm10": data["air"]["PM10_now"],
            "station": data["air"]["station"],
            "station_code": data["station_info"]["측정소코드"],
            "network_type": data["station_info"]["망"],
            "requested_region": region,
            "data_time": data["air"]["dataTime"],
            "feature_timestamp": data["timestamp"],
            "model_input_features": data["df"].iloc[0].to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("예측 실패: %s", e)
        raise HTTPException(status_code=500, detail=f"예측 처리 실패: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)