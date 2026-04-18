from fastapi import FastAPI
import joblib
import pandas as pd
import requests
import datetime
import os
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="실시간 미세먼지(PM10) 예측 서비스")

# ---------------------------------------------------------
# 1. 모델 및 전처리 파일 로드 (경로를 실제 파일 위치로 확인하세요!)
# ---------------------------------------------------------
print("📦 최정예 모델 및 스케일러 로딩 중...")

try:
    # 주호님이 저장하신 '20개 변수 전용' 파일들입니다.
    xgb_model = joblib.load("./dust_xgb_model_final.pkl")
    scaler = joblib.load("./dust_scaler_final.pkl")
    encoder = joblib.load("./dust_encoder_final.pkl")
    
    # 🌟 핵심: 다이어트 성공한 20개 컬럼 리스트 (순서가 학습 때와 같아야 합니다)
    feature_columns = [
        'prev_pm10_1h', 'prev_pm10_2h', 'PM25', 'Season_Spring', '3시간신적설(cm)',
        '적설(cm)', '습도(%)', '현상번호(국내식)', 'prev_pm10_3h', 'precip_lag1',
        'Season_Autumn', 'precip_lag2', 'Month_2', '30cm 지중온도(°C)', 'Season_Summer',
        'region_서울 관악구', 'region_서울 중구', '20cm 지중온도(°C)', 'Month_11', 'NO2'
    ]
    print(f"✅ 로드 완료! (사용 변수: {len(feature_columns)}개)")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    xgb_model = scaler = encoder = None

KMA_API_KEY = os.environ.get("KMA_API_KEY")
AIR_API_KEY = os.environ.get("AIR_API_KEY")

# ---------------------------------------------------------
# 2. 데이터 수집 함수 (기상청 & 에어코리아)
# ---------------------------------------------------------
def get_weather_data(target_dt):
    """기상청 API: 특정 시간의 기상 정보 가져오기"""
    tm = target_dt.strftime("%Y%m%d%H%M")
    url = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?tm={tm}&stn=108&help=1&authKey={KMA_API_KEY}"
    
    try:
        res = requests.get(url, timeout=5)
        lines = [l.strip() for l in res.text.split("\n") if l.strip() and not l.startswith("#")]
        if not lines: return None
        parts = lines[0].split()
        
        # 기상청 데이터 인덱스 매핑 (TA:11, WS:3, HM:13, RN:15, 적설:18 등)
        return {
            "temp": float(parts[11]),
            "wind": float(parts[3]),
            "humid": float(parts[13]),
            "rain": 0.0 if parts[15] in ["-9.0", ""] else float(parts[15]),
            "snow": 0.0 if parts[18] in ["-9.0", ""] else float(parts[18]),
            "snow_3h": 0.0 if parts[17] in ["-9.0", ""] else float(parts[17]), # 3시간 신적설
            "ground_20": float(parts[22]), # 20cm 지중온도
            "ground_30": float(parts[23]), # 30cm 지중온도
            "weather_code": float(parts[14]) # 현상번호
        }
    except: return None

def get_air_data(region):
    """에어코리아 API: 실시간 및 과거 3시간 PM10/PM25 가져오기"""
    station = region.split()[-1] if region else "중구"
    url = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
    params = {"serviceKey": AIR_API_KEY, "returnType": "json", "stationName": station, "dataTerm": "DAILY", "ver": "1.3"}
    
    try:
        res = requests.get(url, params=params, timeout=5).json()
        items = res['response']['body']['items']
        def s_f(v): return float(v) if v and v not in ["-", "null"] else 0.0
        
        return {
            "PM10_now": s_f(items[0]['pm10Value']),
            "PM25": s_f(items[0]['pm25Value']),
            "NO2": s_f(items[0]['no2Value']),
            "prev_pm10_1h": s_f(items[1]['pm10Value']),
            "prev_pm10_2h": s_f(items[2]['pm10Value']),
            "prev_pm10_3h": s_f(items[3]['pm10Value']),
            "dataTime": items[0]['dataTime']
        }
    except: return None

# ---------------------------------------------------------
# 3. 핵심 로직: 입력 데이터프레임 생성
# ---------------------------------------------------------
def build_features(user_region):
    now = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    
    # 데이터 수집 (안전빵으로 현재~2시간 전까지)
    w_now = get_weather_data(now)
    w_1h = get_weather_data(now - datetime.timedelta(hours=1))
    w_2h = get_weather_data(now - datetime.timedelta(hours=2))
    air = get_air_data(user_region)
    
    if not w_now or not air:
        return {"error": "데이터 수집 실패 (기상청/에어코리아 점검 중일 수 있음)"}

    # 학습 때 사용했던 20개 변수 이름에 맞춰 값 매핑
    raw_input = {
        'prev_pm10_1h': air['prev_pm10_1h'],
        'prev_pm10_2h': air['prev_pm10_2h'],
        'PM25': air['PM25'],
        '3시간신적설(cm)': w_now['snow_3h'],
        '적설(cm)': w_now['snow'],
        '습도(%)': w_now['humid'],
        '현상번호(국내식)': w_now['weather_code'],
        'prev_pm10_3h': air['prev_pm10_3h'],
        'precip_lag1': 1 if w_1h and w_1h['rain'] > 0 else 0, # 비 왔었는지 여부
        'precip_lag2': 1 if w_2h and w_2h['rain'] > 0 else 0,
        '30cm 지중온도(°C)': w_now['ground_30'],
        '20cm 지중온도(°C)': w_now['ground_20'],
        'NO2': air['NO2'],
    }

    df = pd.DataFrame([raw_input])
    
    # One-Hot 필드 강제 생성 (0으로 초기화 후 해당되는 것만 1)
    df = df.reindex(columns=feature_columns, fill_value=0.0)
    
    # 계절/달/지역 활성화
    month = now.month
    if month in [3,4,5]: df['Season_Spring'] = 1.0
    elif month in [6,7,8]: df['Season_Summer'] = 1.0
    elif month in [9,10,11]: df['Season_Autumn'] = 1.0
    
    if month == 2: df['Month_2'] = 1.0
    if month == 11: df['Month_11'] = 1.0
    
    if user_region and f"region_{user_region}" in df.columns:
        df[f"region_{user_region}"] = 1.0

    return {"df": df, "info": air, "weather": w_now}

# ---------------------------------------------------------
# 4. API 엔드포인트
# ---------------------------------------------------------
@app.get("/predict")
def predict(region: str = "서울 중구"):
    data = build_features(region)
    if "error" in data: return data
    
    try:
        # 스케일링 -> 예측 -> 역변환(숫자 0,1,2 -> '좋음' 등)
        input_scaled = scaler.transform(data['df'])
        pred = xgb_model.predict(input_scaled)
        grade = encoder.inverse_transform(pred)[0]
        
        return {
            "status": "success",
            "prediction": grade,
            "current_pm10": data['info']['PM10_now'],
            "station": region,
            "data_time": data['info']['dataTime']
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)