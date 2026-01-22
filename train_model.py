# train_model.py
import pandas as pd
import numpy as np
import joblib
import json
import lightgbm as lgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout

# 과제 1 : 가중평균, 표준편차 적용하기 
# 과제 2 : 


# 데이터 로드 (파일 경로 확인 필요)
df = pd.read_csv('K리그_통합데이터_result.csv')

# --- 1. 전처리 ---
row_0 = df.iloc[0]
new_columns = []
for col, val in zip(df.columns, row_0):
    col_clean = str(col).split('.')[0]
    if col == 'Rnd.': col_clean = 'Rnd'
    if str(val) in ['시도', '성공', '성공%']:
        new_columns.append(f"{col_clean}_{str(val)}")
    else:
        new_columns.append(col_clean)

df.columns = new_columns
df = df.drop(0).reset_index(drop=True)
df = df[df['Rnd'] != 'Rnd.']

cols_to_numeric = df.columns.drop(['대회', 'H/A', '팀명', '시즌'])
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Rnd', '득점'])
df = df.sort_values(['시즌', 'Rnd'])

# --- 2. 피처 선정 및 스케일링 ---
features = [
    '득점', '도움', '슈팅', '유효 슈팅', 'PA내 슈팅',
    '패스_성공%', '키패스', '공격진영 패스_성공',
    '경합 지상_성공%', '경합 공중_성공%', 
    '인터셉트', '차단', '파울'
]

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# --- 3. 데이터셋 생성 ---
def create_dataset(df, window_size=5):
    X_home_seq, X_away_seq, X_2d, y = [], [], [], []
    
    matches = pd.merge(
        df[df['H/A'] == 'HOME'],
        df[df['H/A'] == 'AWAY'],
        on=['시즌', 'Rnd', '대회'],
        suffixes=('', '_opp')
    )
    
    for _, row in matches.iterrows():
        home_team, away_team = row['팀명'], row['팀명_opp']
        season, rnd = row['시즌'], row['Rnd']
        
        home_hist = df[(df['팀명'] == home_team) & (df['시즌'] == season) & (df['Rnd'] < rnd)].tail(window_size)
        away_hist = df[(df['팀명'] == away_team) & (df['시즌'] == season) & (df['Rnd'] < rnd)].tail(window_size)
        
        if len(home_hist) == window_size and len(away_hist) == window_size:
            X_home_seq.append(home_hist[features].values)
            X_away_seq.append(away_hist[features].values)
            home_mean = home_hist[features].mean().values
            away_mean = away_hist[features].mean().values
            X_2d.append(np.concatenate([home_mean, away_mean, [1]])) 
            # 승 = 2 / 무 = 1 / 패 = 0
            if row['득점'] > row['득점_opp']: target = 2
            elif row['득점'] < row['득점_opp']: target = 0
            else: target = 1
            y.append(target)
            
    return np.array(X_home_seq), np.array(X_away_seq), np.array(X_2d), np.array(y)

X_h, X_a, X_2d, y = create_dataset(df)

X_h_tr, X_h_te, X_a_tr, X_a_te, X_2d_tr, X_2d_te, y_tr, y_te = train_test_split(
    X_h, X_a, X_2d, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. 모델 학습 ---
print(">>> LightGBM 학습 중...") # lightGBM -> 2D
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, verbosity=-1)
lgb_model.fit(X_2d_tr, y_tr)



print(">>> LSTM 학습 중...") # LSTM -> 3D
# home away 별로 lstm 별도로 구하고 Concatenate() 사용
# 이 모델 정확도 높이려면 merged x 를 Dense 층을 추가해줘야하나? 
# feature 수 확인
input_h = Input(shape=(5, len(features)), name='Home_Input')
lstm_h = LSTM(32)(input_h)

input_a = Input(shape=(5, len(features)), name='Away_Input')
lstm_a = LSTM(32)(input_a)

merged = Concatenate()([lstm_h, lstm_a])
x = Dense(64, activation='relu')(merged) # 뉴런 수 늘림
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)      # 층 추가
x = Dropout(0.2)(x)
output = Dense(3, activation='softmax')(x)

lstm_model = Model(inputs=[input_h, input_a], outputs=output)
lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit([X_h_tr, X_a_tr], y_tr, epochs=30, batch_size=16, verbose=0)

# --- 5. 저장 (경로 제거: 현재 폴더에 저장) ---
print(">>> 파일 저장 중 (현재 폴더)...")
joblib.dump(lgb_model, 'lgb_model.pkl')
lstm_model.save('lstm_model.keras')
joblib.dump(scaler, 'scaler.pkl')

latest_data = {}
for team in df['팀명'].unique():
    last_5 = df[df['팀명'] == team].sort_values(['시즌', 'Rnd']).tail(5)[features].values
    if len(last_5) == 5:
        latest_data[team] = last_5.tolist()

with open('team_recent_data.json', 'w', encoding='utf-8') as f:
    json.dump(latest_data, f, ensure_ascii=False)

print(">>> 완료!")