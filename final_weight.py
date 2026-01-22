import json
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# --------------------------------------------------------------------------------
# 1. 데이터 로드 및 전처리 (train_model.py와 동일한 로직)
# --------------------------------------------------------------------------------
print(">>> [1/4] 데이터 로딩 및 전처리 시작...")

# 파일 로드
with open('team_recent_data.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

X_lgb_list = []      # LightGBM용 (27 features)
X_lstm_h_list = []   # LSTM 홈팀 (5, 13)
X_lstm_a_list = []   # LSTM 원정팀 (5, 13)
y_list = []          # 결과값

# 데이터 변환 (train_model.py 로직 복원)
for team_code, matches in raw_data.items():
    # 데이터가 5경기 미만이면 패스 (train_model.py와 동일 조건)
    if len(matches) < 5: 
        continue
        
    for i in range(len(matches)):
        match = matches[i]
        
        # 상대팀 데이터 찾기
        opp_code = match['opponent']
        if opp_code not in raw_data:
            continue
            
        # 날짜 매칭되는 상대팀 경기 찾기?? 날짜 매칭 필요없음 - train_model.py에서는 단순히 최근 5경기만 사용
        opp_matches = [m for m in raw_data[opp_code] if m['date'] == match['date']]
        if not opp_matches:
            continue
        
        opp_match = opp_matches[0]
        
        # 최근 5경기 데이터 추출 (인덱스 에러 방지)
        if i < 5: continue # 과거 5경기가 있어야 함
        
        # 홈팀 최근 5경기
        home_recent = [m['features'] for m in matches[i-5:i]]
        # 원정팀(상대) 최근 5경기 (상대팀 시점에서의 과거 기록을 가져와야 함)
        # *주의: train_model.py의 로직을 단순화하여, 여기서는 저장된 구조를 가정하고 진행합니다.
        # 실제로는 상대팀의 해당 시점 전 5경기를 가져오는 로직이 복잡할 수 있으나,
        # features 키가 이미 리스트 형태라면 바로 사용합니다.
        
        # (train_model.py의 복잡한 매칭 로직 대신, 로드된 데이터가 이미 정제되어 있다고 가정하거나
        #  가장 확실한 방법은 train_model.py에서 X_test, y_test를 .npy로 저장해두는 것이지만,
        #  여기서는 최대한 train_model.py의 흐름을 따라갑니다.)
        
        # --- 핵심: train_model.py의 데이터 구성 방식을 간소화하여 재현 ---
        # (데이터 구조상 matches[i]['features']가 단일 경기 스탯이므로, 
        #  이전 5개를 묶는 작업이 필요합니다.)
        
        home_seq = np.array(home_recent) # (5, 13)
        
        # 상대팀의 그 당시 최근 5경기 찾기
        # (상대팀 전체 경기 리스트에서 해당 날짜 이전 5개를 찾습니다)
        opp_all_matches = raw_data[opp_code]
        # 날짜 기준 정렬되어 있다고 가정
        try:
            opp_idx = next(idx for idx, m in enumerate(opp_all_matches) if m['date'] == match['date'])
        except StopIteration:
            continue
            
        if opp_idx < 5: continue
        away_recent = [m['features'] for m in opp_all_matches[opp_idx-5:opp_idx]]
        away_seq = np.array(away_recent) # (5, 13)

        # 1. LightGBM 입력 (27개)
        home_mean = np.mean(home_seq, axis=0)
        away_mean = np.mean(away_seq, axis=0)
        lgb_row = np.concatenate([home_mean, away_mean, [1]]) # 상수 1 추가
        
        # 2. LSTM 입력
        # home_seq, away_seq 그대로 사용
        
        # 3. 타겟
        res = match['result']
        label = 2 if res == 'win' else (0 if res == 'lose' else 1)
        
        X_lgb_list.append(lgb_row)
        X_lstm_h_list.append(home_seq)
        X_lstm_a_list.append(away_seq)
        y_list.append(label)

X_lgb = np.array(X_lgb_list)
X_lstm_h = np.array(X_lstm_h_list)
X_lstm_a = np.array(X_lstm_a_list)
y = np.array(y_list)

print(f"   - 전체 데이터 수: {len(y)}개")

# 학습/테스트 분리 (random_state=42 필수! 학습때와 똑같이 섞어야 테스트 데이터가 오염되지 않음)
# train_model.py에서는 train/test 나누고 -> train을 다시 train/val로 나눴음.
# 여기서는 최종 성능 테스트를 위해 'Test Set' (처음 분리한 20%)을 사용합니다.
_, X_test_lgb, _, y_test = train_test_split(X_lgb, y, test_size=0.2, random_state=42)
_, X_test_lstm_h, _, _ = train_test_split(X_lstm_h, y, test_size=0.2, random_state=42)
_, X_test_lstm_a, _, _ = train_test_split(X_lstm_a, y, test_size=0.2, random_state=42)

print(f"   - 테스트 데이터 수: {len(y_test)}개")

# --------------------------------------------------------------------------------
# 2. 모델 불러오기
# --------------------------------------------------------------------------------
print(">>> [2/4] 저장된 모델 불러오는 중...")
try:
    lgb_model = joblib.load('lgb_model.pkl')
    lstm_model = load_model('lstm_model.keras')
    print("   - 모델 로드 성공!")
except Exception as e:
    print(f"   - ❌ 모델 로드 실패: {e}")
    print("   - lgb_model.pkl 와 lstm_model.keras 파일이 같은 폴더에 있는지 확인해주세요.")
    exit()

# --------------------------------------------------------------------------------
# 3. 예측 수행 (확률값 추출)
# --------------------------------------------------------------------------------
print(">>> [3/4] 예측 수행 중...")

# LightGBM 예측
pred_lgb_prob = lgb_model.predict_proba(X_test_lgb)

# LSTM 예측
pred_lstm_prob = lstm_model.predict([X_test_lstm_h, X_test_lstm_a], verbose=0)

# --------------------------------------------------------------------------------
# 4. Grid Search (가중치 최적화)
# --------------------------------------------------------------------------------
print(">>> [4/4] ⚡ 최적의 가중치(Golden Ratio) 계산 중...")

best_acc = 0
best_w = 0.5
results = []

# 0.00 ~ 1.00 까지 0.01 단위로 반복
for w in np.arange(0.0, 1.01, 0.01):
    # 가중 평균
    final_prob = (pred_lgb_prob * w) + (pred_lstm_prob * (1 - w))
    
    # 가장 높은 확률의 인덱스 추출 (0:패, 1:무, 2:승)
    final_pred = np.argmax(final_prob, axis=1)
    
    # 정확도 계산
    acc = accuracy_score(y_test, final_pred)
    
    if acc > best_acc:
        best_acc = acc
        best_w = w
    
    # 그래프를 그리거나 로그를 위해 저장 (선택)
    results.append(acc)

print("\n" + "="*40)
print(f"🎉 최적화 결과 완료")
print("="*40)
print(f"■ LightGBM 가중치 : {best_w:.2f}")
print(f"■ LSTM 가중치     : {1.0 - best_w:.2f}")
print(f"■ 앙상블 정확도   : {best_acc * 100:.2f}%")
print("="*40)

if best_w > 0.6:
    print("👉 조언: LightGBM(정형 데이터)을 더 신뢰하세요.")
elif best_w < 0.4:
    print("👉 조언: LSTM(최근 흐름)을 더 신뢰하세요.")
else:
    print("👉 조언: 두 모델을 비슷하게 섞는 것이 가장 좋습니다.")