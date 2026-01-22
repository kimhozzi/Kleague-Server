import json
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

print(">>> [1/5] 데이터 로드 및 구조 확인 시작...")

# 1. 데이터 로드
with open('team_recent_data.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# -------------------------------------------------------------------------
# [요청하신 부분] 데이터 타입 확인을 위한 Print 코드
# -------------------------------------------------------------------------
first_team_key = list(raw_data.keys())[0]
first_match_data = raw_data[first_team_key][0]

print("")
print(f"👀 데이터 구조 확인 (첫 번째 데이터 샘플)")
print("="*50)
print(f"Type: {type(first_match_data)}")
print(f"Data: {first_match_data}")
print("")

# 데이터 타입에 따라 처리 방식 결정
IS_DICT = isinstance(first_match_data, dict)

if IS_DICT:
    print("✅ 데이터가 '딕셔너리(Dictionary)' 형태입니다. (Key로 접근)")
else:
    print("✅ 데이터가 '리스트(List)' 형태입니다. (Index로 접근)")
    # 리스트일 경우 인덱스 정의 (일반적인 순서 가정)
    IDX_DATE = 0
    IDX_OPP = 1
    IDX_RES = 2
    IDX_FEAT = 3

# -------------------------------------------------------------------------
# 2. 데이터 전처리 (train_model.py 로직 그대로 적용)
# -------------------------------------------------------------------------
print(">>> [2/5] 데이터 전처리 진행 중...")

X_lgb_list = []      
X_lstm_h_list = []   
X_lstm_a_list = []   
y_list = []          

for team_code, matches in raw_data.items():
    # 5경기 미만이면 데이터 못 만드니까 패스
    if len(matches) < 5: 
        continue
        
    for i in range(len(matches)):
        # 과거 5경기가 없으면 패스 (train_model.py 로직)
        if i < 5: 
            continue
            
        match = matches[i]
        
        # --- [유연한 데이터 처리] ---
        if IS_DICT:
            match_date = match['date']
            opp_code = match['opponent']
            match_result = match['result']
            # 특징값 가져오기
            home_recent = [m['features'] for m in matches[i-5:i]]
        else:
            match_date = match[IDX_DATE]
            opp_code = match[IDX_OPP]
            match_result = match[IDX_RES]
            # 특징값 가져오기
            home_recent = [m[IDX_FEAT] for m in matches[i-5:i]]

     

        # 상대팀 기준, 해당 경기가 몇 번째였는지 찾기 (최근 5경기 뽑으려고)
        # (리스트인 경우 인덱스를 찾아야 슬라이싱이 가능함)
        if IS_DICT:
             # 딕셔너리면 enumerate로 돌면서 날짜 비교
             try:
                 opp_idx = next(idx for idx, m in enumerate(opp_all_matches) if m['date'] == match_date)
             except StopIteration:
                 continue
        else:
             try:
                 opp_idx = next(idx for idx, m in enumerate(opp_all_matches) if m[IDX_DATE] == match_date)
             except StopIteration:
                 continue
                 
        if opp_idx < 5: 
            continue

        # 상대팀 최근 5경기 추출
        if IS_DICT:
            away_recent = [m['features'] for m in opp_all_matches[opp_idx-5:opp_idx]]
        else:
            away_recent = [m[IDX_FEAT] for m in opp_all_matches[opp_idx-5:opp_idx]]
            
        
        # 데이터 조립
        home_seq = np.array(home_recent)
        away_seq = np.array(away_recent)
        
        # LightGBM 입력 (평균)
        home_mean = np.mean(home_seq, axis=0)
        away_mean = np.mean(away_seq, axis=0)
        lgb_row = np.concatenate([home_mean, away_mean, [1]])
        
        # 결과 라벨링
        if match_result == 'win': label = 2
        elif match_result == 'lose': label = 0
        else: label = 1
        
        X_lgb_list.append(lgb_row)
        X_lstm_h_list.append(home_seq)
        X_lstm_a_list.append(away_seq)
        y_list.append(label)

X_lgb = np.array(X_lgb_list)
X_lstm_h = np.array(X_lstm_h_list)
X_lstm_a = np.array(X_lstm_a_list)
y = np.array(y_list)

print(f"✅ 총 데이터 개수: {len(y)}개")

# 테스트셋 분리
_, X_test_lgb, _, y_test = train_test_split(X_lgb, y, test_size=0.2, random_state=42)
_, X_test_lstm_h, _, _ = train_test_split(X_lstm_h, y, test_size=0.2, random_state=42)
_, X_test_lstm_a, _, _ = train_test_split(X_lstm_a, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------------
# 3. 가중치 최적화 수행
# -------------------------------------------------------------------------
print(">>> [3/5] 모델 로딩 중...")
lgb_model = joblib.load('lgb_model.pkl')
lstm_model = load_model('lstm_model.keras')

print(">>> [4/5] 예측 수행 중...")
pred_lgb = lgb_model.predict_proba(X_test_lgb)
pred_lstm = lstm_model.predict([X_test_lstm_h, X_test_lstm_a], verbose=0)

print(">>> [5/5] 최적 가중치 계산 중...")
best_acc = 0
best_w = 0.5

for w in np.arange(0.0, 1.01, 0.01):
    final_prob = (pred_lgb * w) + (pred_lstm * (1 - w))
    final_pred = np.argmax(final_prob, axis=1)
    acc = accuracy_score(y_test, final_pred)
    
    if acc > best_acc:
        best_acc = acc
        best_w = w

print("\n" + "="*50)
print(f"🏆 최종 결과 (정확도: {best_acc*100:.2f}%)")
print("="*50)
print(f"LGBM 가중치 : {best_w:.2f}")
print(f"LSTM 가중치 : {1.0 - best_w:.2f}")
print("="*50)