# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os

app = FastAPI()

# ----------------------------------------------------------------
# 1. CORS 설정 (React 연동)
# ----------------------------------------------------------------
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------
# 2. 전역 변수 및 매핑 정보
# ----------------------------------------------------------------
artifacts = {}

# [중요] React에서 오는 '한글 팀명'을 '데이터 키(영어 약어)'로 바꾸기 위한 사전
KOREAN_TO_CODE = {
    "울산": "ULS", "수원삼성": "SSB", "포항": "POH", "제주": "JEJ",
    "전북": "JEO", "성남": "SNG", "서울": "SEO", "대구": "DAE",
    "인천": "INC", "강원": "GAN", "광주": "GWA", "수원FC": "SFC",
    "김천": "GIM", "대전": "DJN"
}

@app.on_event("startup")
def load_artifacts():
    global artifacts
    print(">>> [System] 모델 및 데이터 로딩 시작...")
    try:
        # 모델 파일 로드
        artifacts['lgb'] = joblib.load('lgb_model.pkl')
        artifacts['lstm'] = load_model('lstm_model.keras')
        
        # 데이터 JSON 로드
        with open('team_recent_data.json', 'r', encoding='utf-8') as f:
            artifacts['stats'] = json.load(f)
            
        print("✅ [System] 모델 및 데이터 로딩 완료! API 서버 준비 끝.")
    except Exception as e:
        print(f"❌ [Error] 로딩 실패: {e}")

# ----------------------------------------------------------------
# 3. API 요청/응답 모델
# ----------------------------------------------------------------
class PredictRequest(BaseModel):
    home_team: str  # React에서 "울산"이라고 보냄
    away_team: str  # React에서 "포항"이라고 보냄

# ----------------------------------------------------------------
# 4. 예측 로직   -- 이거 트러블 슈팅 기록 : 1. 특성 수 불일치 문제 해결(어...괜히 flatten해서 특성 수가 130개가 되어버림) 
# 2. 요청 데이터 변수 명이 맞지 않음 - 한글/영어 매핑 문제 해결 
# 3. 예외 처리 강화 작성한 코드 부분에 주석으로 표시
# ----------------------------------------------------------------
# main.py 의 predict_match 함수 내부 수정

@app.post("/api/predict")
async def predict_match(req: PredictRequest):
    home_name = req.home_team
    away_name = req.away_team
    
    # 1. 한글/영어 매핑
    home_code = KOREAN_TO_CODE.get(home_name, home_name)
    away_code = KOREAN_TO_CODE.get(away_name, away_name)

    stats = artifacts.get('stats', {})

    # 2. 키 찾기 (없으면 에러)
    home_key = home_code if home_code in stats else (home_name if home_name in stats else None)
    away_key = away_code if away_code in stats else (away_name if away_name in stats else None)

    if not home_key:
        raise HTTPException(status_code=404, detail=f"홈 팀 '{home_name}' 데이터 없음")
    if not away_key:
        raise HTTPException(status_code=404, detail=f"원정 팀 '{away_name}' 데이터 없음")

    try:
        # 데이터 가져오기
        home_seq = np.array(stats[home_key]) # (5, 13)
        away_seq = np.array(stats[away_key]) # (5, 13)

        # ---------------------------------------------------------
        # [수정된 부분] 학습 코드(train_model.py)와 로직 통일
        # ---------------------------------------------------------
        
        # 1. LightGBM 입력: (홈평균 + 원정평균 + 상수1) = 13 + 13 + 1 = 27개
        # 기존 코드(flatten)는 130개를 만들어서 에러가 났던 것임! - 평탄화작업을 하지 않았을 시 모델의 정확도 비교
        input_lgb = np.concatenate([
            np.mean(home_seq, axis=0), 
            np.mean(away_seq, axis=0), 
            [1] # 학습 때 넣었던 상수
        ]).reshape(1, -1)

        # 2. LSTM 입력: (1, 5, 13)
        input_lstm_h = home_seq.reshape(1, 5, -1)
        input_lstm_a = away_seq.reshape(1, 5, -1)

        # ---------------------------------------------------------
        
        # 디버깅 로그
        print(f"🤖 LGBM 입력 개수: {input_lgb.shape[1]} (기대값: 27)")

        # 예측
        lgb_prob = artifacts['lgb'].predict_proba(input_lgb)[0]
        lstm_prob = artifacts['lstm'].predict([input_lstm_h, input_lstm_a], verbose=0)[0]

        # 앙상블
        # 두 모델 동일 비중 vs 가중치 부여 정확도의 비교
        # 파일 하나 생성 후 최적의 가중치 탐색 예정 
        final_prob = (lgb_prob * 0.5) + (lstm_prob * 0.5)
        
        idx = np.argmax(final_prob)
        pred_text = "승 (Win)" if idx == 2 else ("패 (Loss)" if idx == 0 else "무 (Draw)")

        return {
            "home_team": home_name,
            "away_team": away_name,
            "prediction": pred_text,
            "probability": {
                "win": float(final_prob[2]),
                "draw": float(final_prob[1]),
                "lose": float(final_prob[0])
            }
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"서버 에러: {str(e)}")



#     문제의 원인은: 이 코드(train_model)를 **실행(Run)**하지 않아서, team_recent_data.json 파일이 옛날 버전(모든 컬럼 130개가 다 들어있는 상태)으로 남아있기 때문입니다.

# 👉 지금 바로 터미널에서 train_model.py를 실행해주세요.

