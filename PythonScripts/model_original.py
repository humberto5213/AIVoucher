# LSTM 모델 코드 
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout 
import keras.backend as K
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
import tensorflow as tf 
from collections import defaultdict
import pickle
import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# lstm unit 수나 dropout rate 등도 하이퍼파라미터
def LSTM_MODEL(selected_features, window_size):
  K.clear_session()
  grid_model = Sequential()
  grid_model.add(LSTM(64,return_sequences=True,input_shape=(window_size, len(selected_features))))
  #grid_model.add(Dropout(0.2))
  grid_model.add(LSTM(32))
  grid_model.add(Dropout(0.2))
  grid_model.add(Dense(16))
  grid_model.add(Dense(len(selected_features)))
  #grid_model.summary() 
  return grid_model 

# lstm 모델에 input으로 시계열을 넣어주기 위해 전처리하는 함수 
def make_lstm_input(df, past_steps, future_steps): # train or test dataframe ((data수,feature수)) / 24 / 1 
  # lstm input 형식을 위해 3차원으로 재구성하기 (sample수, time_steps(=24), features(=4))
  x = df.values[np.arange(past_steps)[None, :] + np.arange(df.shape[0]-past_steps)[:, None]]
  y = df.values[np.arange(future_steps)[None, :] + np.arange(df.shape[0])[past_steps:, None]]
  return x, y 

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

def train(X_dict, Y_dict, org, selected_features, window_size, lr, ep, bs, train_split_ratio):
  for k, x_df in X_dict.items(): 
    print(k, ' 데이터 전처리 중..')
    train_len = int(len(x_df) * train_split_ratio)
    X_train_df = x_df.iloc[:train_len,:]
    X_test_df = x_df.iloc[train_len:,:]

    y_df = Y_dict[k]
    Y_train_df = y_df.iloc[:train_len,:]
    Y_test_df = y_df.iloc[train_len:,:]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(X_train_df)
    y_scaler.fit(Y_train_df)
    
    X_train = x_scaler.transform(X_train_df)
    X_train_df = pd.DataFrame(X_train, columns=X_train_df.columns, index=list(X_train_df.index.values))

    X_test = x_scaler.transform(X_test_df)
    X_test_df = pd.DataFrame(X_test, columns=X_test_df.columns, index=list(X_test_df.index.values))

    Y_train = y_scaler.transform(Y_train_df)
    Y_train_df = pd.DataFrame(Y_train, columns=Y_train_df.columns, index=list(Y_train_df.index.values))

    Y_test = y_scaler.transform(Y_test_df)
    Y_test_df = pd.DataFrame(Y_test, columns=Y_test_df.columns, index=list(Y_test_df.index.values))

    # lstm모델에 input으로 넣기위한 전처리 
    X_train_lstm, _ = make_lstm_input(X_train_df, window_size, 1)
    _, Y_train_lstm = make_lstm_input(Y_train_df, window_size, 1) # 국가망 데이터를 통한 보정 
    
    X_test_lstm, _ = make_lstm_input(X_test_df, window_size, 1)
    _, Y_test_lstm = make_lstm_input(Y_test_df, window_size, 1) # 테스트 시 성능 평가는 국가망 (Y_test) 데이터 기준 

    print(k, '학습 시작..')
    # lstm 학습 
    lstm_model = LSTM_MODEL(selected_features, window_size)
    ##############################Hyper Parameter##############################
    # learning rate, epochs, batch_size 등 하이퍼파라미터 위에서 설정하였음 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    lstm_model.compile(optimizer=opt, loss='mse')

    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    lstm_model.fit(X_train_lstm, Y_train_lstm, epochs=ep, batch_size=bs, verbose=1, callbacks=[early_stop, lr_scheduler], shuffle=False)

    print(k, '학습 완료..')
    #학습된 lstm 저장 
    lstm_model.save('./trained_models/{}_trained_models/{}_{}_trained_lstm'.format(org, k, match_dict[k]))
  

def evaluate(X_dict, Y_dict, org, selected_features, window_size, lr, ep, bs, train_split_ratio):
  train_eval_dict = dict()
  test_eval_dict = dict() 
  for k, x_df in X_dict.items(): # [0] 빼야함 ############
    print(k,' 데이터 전처리 중..')
    train_len = int(len(x_df) * train_split_ratio)
    X_train_df = x_df.iloc[:train_len,:]
    X_test_df = x_df.iloc[train_len:,:]

    y_df = Y_dict[k]
    Y_train_df = y_df.iloc[:train_len,:]
    Y_test_df = y_df.iloc[train_len:,:]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(X_train_df)
    y_scaler.fit(Y_train_df)
    
    X_train = x_scaler.transform(X_train_df)
    X_train_df = pd.DataFrame(X_train, columns=X_train_df.columns, index=list(X_train_df.index.values))

    X_test = x_scaler.transform(X_test_df)
    X_test_df = pd.DataFrame(X_test, columns=X_test_df.columns, index=list(X_test_df.index.values))

    Y_train = y_scaler.transform(Y_train_df)
    Y_train_df = pd.DataFrame(Y_train, columns=Y_train_df.columns, index=list(Y_train_df.index.values))

    Y_test = y_scaler.transform(Y_test_df)
    Y_test_df = pd.DataFrame(Y_test, columns=Y_test_df.columns, index=list(Y_test_df.index.values))

    # lstm모델에 input으로 넣기위한 전처리 
    X_train_lstm, _ = make_lstm_input(X_train_df, window_size, 1)
    _, Y_train_lstm = make_lstm_input(Y_train_df, window_size, 1) # 국가망 데이터를 통한 보정 
    
    X_test_lstm, _ = make_lstm_input(X_test_df, window_size, 1)
    _, Y_test_lstm = make_lstm_input(Y_test_df, window_size, 1) # 테스트 시 성능 평가는 국가망 (Y_test) 데이터 기준 

    ######################################위 전처리까지는 동일
    # 학습된 lstm 모델 load
    print(k, '학습된 모델 load 및 test 시작..')
    
    with open('{}_match_dict.pickle'.format(org), 'rb') as handle:
      match_dict = pickle.load(handle)   
    print('match_dict load 완료..')
    try:      
      lstm_model = tf.keras.models.load_model('./trained_models/{}_trained_models/{}_{}_trained_lstm'.format(org, k, match_dict[k]))
    except: 
      print('Test 실패. 학습을 먼저 진행하여 trained_models에 모델 weight 파일이 준비되어야 합니다.')
    ##############################Hyper Parameter##############################
    # learning rate, epochs, batch_size 등 하이퍼파라미터 위에서 설정하였음 
    train_pred = lstm_model.predict(X_train_lstm)
    test_pred = lstm_model.predict(X_test_lstm)

    train_MAPE = mean_absolute_error(Y_train_lstm.squeeze(), train_pred)
    test_MAPE = mean_absolute_error(Y_test_lstm.squeeze(), test_pred)
    train_RMSE = mean_squared_error(Y_train_lstm.squeeze(), train_pred)
    test_RMSE = mean_squared_error(Y_test_lstm.squeeze(), test_pred)
    train_R2 = r2_score(Y_train_lstm.squeeze(), train_pred)
    test_R2 = r2_score(Y_test_lstm.squeeze(), test_pred)
    print('{}_{}_TRAIN ~ MAPE, RMSE, R2 {}, {}, {} :'.format(k,match_dict[k], train_MAPE, train_RMSE, train_R2))
    print('{}_{}_TEST ~ MAPE, RMSE, R2 {}, {}, {} :'.format(k,match_dict[k], test_MAPE, test_RMSE, test_R2))
    train_eval_dict[k] = [train_MAPE, train_RMSE, train_R2] # train_MAPE, train_RMSE, train_R2 순 
    test_eval_dict[k] = [test_MAPE, test_RMSE, test_R2] # test_MAPE, test_RMSE, test_R2 순
  
  print('trainset, testset에 대한 성능 평가 pickle 파일 저장 중..')
  with open('{}_train_eval_dict.pickle'.format(org), 'wb') as handle:
    pickle.dump(train_eval_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

  with open('{}_test_eval_dict.pickle'.format(org), 'wb') as handle:
    pickle.dump(test_eval_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
  
  # 전 지역에 대한 평균 평가 결과 (train, test 각각에 대한)
  train_avg = [0,0,0]
  for k, v in train_eval_dict.items():
    train_avg[0] += v[0]
    train_avg[1] += v[1]
    train_avg[2] += v[2]
  for i,v in enumerate(train_avg):
    train_avg[i] /= len(train_eval_dict)

  test_avg = [0,0,0]
  for k, v in test_eval_dict.items():
    test_avg[0] += v[0]
    test_avg[1] += v[1]
    test_avg[2] += v[2]
  for i,v in enumerate(test_avg):
    test_avg[i] /= len(test_eval_dict)

  print('train_average_eval (MAPE, RMSE, R2): ' , train_avg)
  print('test_average_eval (MAPE, RMSE, R2): ' , test_avg)