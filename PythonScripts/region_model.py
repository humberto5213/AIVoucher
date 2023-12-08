# region_model.py 
# one region 용 
# LSTM 모델 코드 
# time series cross validation 개념 참조 : https://medium.com/keita-starts-data-science/time-series-split-with-scikit-learn-74f5be38489e
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout 
import keras.backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping
import tensorflow as tf 
from collections import defaultdict
import pickle
import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from copy import deepcopy
from sklearn.model_selection import TimeSeriesSplit

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

    x_scaler = StandardScaler() #MinMaxScaler() #StandardScaler()
    y_scaler = StandardScaler() #MinMaxScaler() #StandardScaler()
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

    with open('{}_match_dict.pickle'.format(org), 'rb') as handle:
      match_dict = pickle.load(handle)   
    print('match_dict load 완료..')

    print(k, '학습 시작..')
    # lstm 학습 
    lstm_model = LSTM_MODEL(selected_features, window_size)
    ##############################Hyper Parameter##############################
    # learning rate, epochs, batch_size 등 하이퍼파라미터 위에서 설정하였음 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    lstm_model.compile(optimizer=opt, loss='mse')

    early_stop = EarlyStopping(monitor='loss', patience=30, verbose=1)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    lstm_model.fit(X_train_lstm, Y_train_lstm, epochs=ep, batch_size=bs, verbose=1, callbacks=[early_stop, lr_scheduler], shuffle=False)

    print(k, '학습 완료..')
    #학습된 lstm 저장 

    print(k, ' 지역에 대해 학습된 모델 저장 중.. @ <trained_models/one_trained_model>')
    lstm_model.save('./trained_models/one_trained_model/{}_{}_trained_lstm'.format(k, match_dict[k]))

def evaluate(X_dict, Y_dict, org, selected_features, window_size, lr, ep, bs, train_split_ratio):
  train_eval_dict = dict()
  test_eval_dict = dict() 
  preds_dict = dict() 

  for k, x_df in X_dict.items():
    print(k,' 데이터 전처리 중..')
    train_len = int(len(x_df) * train_split_ratio)
    X_train_df = x_df.iloc[:train_len,:]
    X_test_df = x_df.iloc[train_len:,:]

    y_df = Y_dict[k]
    Y_train_df = y_df.iloc[:train_len,:]
    Y_test_df = y_df.iloc[train_len:,:]

    x_scaler = StandardScaler()#MinMaxScaler() #StandardScaler()
    y_scaler = StandardScaler()#MinMaxScaler() #StandardScaler()
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
    print('match_dict load 완료..\n')
    try:      
      lstm_model = tf.keras.models.load_model('./trained_models/one_trained_model/{}_{}_trained_lstm'.format(k, match_dict[k]))
    except: 
      print('Test 실패. 학습을 먼저 진행하여 trained_models에 모델 weight 파일이 준비되어야 합니다.')
    ##############################Hyper Parameter##############################
    # learning rate, epochs, batch_size 등 하이퍼파라미터 위에서 설정하였음 
    train_pred = lstm_model.predict(X_train_lstm)
    test_pred = lstm_model.predict(X_test_lstm)
    preds_dict[k] = (train_pred, test_pred)

    # predicted values 를 csv로 저장 
    preds_df = deepcopy(x_df)
    preds_df.iloc[ : window_size] = None # 처음 window_size 만큼은 예측값 없음 
    preds_df.iloc[window_size : len(train_pred)+window_size] = train_pred
    preds_df.iloc[len(train_pred)+window_size : len(train_pred)+window_size+window_size] = None
    preds_df.iloc[len(train_pred)+window_size+window_size : ] = test_pred
    
    inverse_scaled_preds_df = x_scaler.inverse_transform(preds_df) # scaled 된 결과가 예측되므로 inverse_transform 해줘야함 
    preds_df = deepcopy(x_df) #df 자리만 만듦. 
    preds_df.iloc[:] = inverse_scaled_preds_df
    preds_df.to_csv('one_region_predicted_value/{}_{}_predicted_value.csv'.format(k, match_dict[k]))

    train_MAE = mean_absolute_error(Y_train_lstm.squeeze(), train_pred)
    test_MAE = mean_absolute_error(Y_test_lstm.squeeze(), test_pred)

    train_MAPE = mean_absolute_percentage_error(Y_train_lstm.squeeze(), train_pred)
    test_MAPE = mean_absolute_percentage_error(Y_test_lstm.squeeze(), test_pred)

    train_RMSE = mean_squared_error(Y_train_lstm.squeeze(), train_pred)
    test_RMSE = mean_squared_error(Y_test_lstm.squeeze(), test_pred)

    train_R2 = r2_score(Y_train_lstm.squeeze(), train_pred)
    test_R2 = r2_score(Y_test_lstm.squeeze(), test_pred)
    print('{}_{}_TRAIN ~ MAE, MAPE, RMSE, R2 {}, {}, {}, {} '.format(k,match_dict[k], train_MAE, train_MAPE, train_RMSE, train_R2))
    print('{}_{}_TEST ~ MAE, MAPE, RMSE, R2 {}, {}, {}, {} '.format(k,match_dict[k], test_MAE, test_MAPE, test_RMSE, test_R2))
    train_eval_dict[k] = [train_MAE, train_MAPE, train_RMSE, train_R2] # train_MAE,train_MAPE, train_RMSE, train_R2 순 
    test_eval_dict[k] = [test_MAE, test_MAPE, test_RMSE, test_R2] # test_MAE, test_MAPE, test_RMSE, test_R2 순
    return train_eval_dict[k], test_eval_dict[k]



def cross_validate(X_dict, Y_dict, org, selected_features, window_size, lr, ep, bs, n_split):
  train_eval_dict = dict()
  test_eval_dict = dict() 
  preds_dict = dict() 
  tscv = TimeSeriesSplit(n_splits=n_split)
  print(tscv)
  print('Start cross validation!')
  cnt = 0 
  train_scores = [] 
  test_scores = [] 
  for k, x_df in X_dict.items(): 
    for train_index, test_index in tscv.split(x_df):
      #print("TRAIN:", train_index, "TEST:", test_index)
      y_df = Y_dict[k]
      X_train_df, X_test_df = x_df.iloc[train_index], x_df.iloc[test_index]
      Y_train_df, Y_test_df = y_df.iloc[train_index], y_df.iloc[test_index]
      print('{} th cross validation ~ (len_X_train, len_X_test) : '.format(cnt), len(X_train_df), len(X_test_df))
      cnt += 1
      x_scaler = StandardScaler() #MinMaxScaler() #StandardScaler()
      y_scaler = StandardScaler() #MinMaxScaler() #StandardScaler()
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

      with open('{}_match_dict.pickle'.format(org), 'rb') as handle:
        match_dict = pickle.load(handle)   
      print('match_dict load 완료..')

      print(k, match_dict[k], '학습 시작..')
      # lstm 학습 
      lstm_model = LSTM_MODEL(selected_features, window_size)
      ##############################Hyper Parameter##############################
      # learning rate, epochs, batch_size 등 하이퍼파라미터 위에서 설정하였음 
      opt = tf.keras.optimizers.Adam(learning_rate=lr)
      lstm_model.compile(optimizer=opt, loss='mse')

      early_stop = EarlyStopping(monitor='loss', patience=30, verbose=1)
      lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
      lstm_model.fit(X_train_lstm, Y_train_lstm, epochs=ep, batch_size=bs, verbose=1, callbacks=[early_stop, lr_scheduler], shuffle=False)
      print(cnt, '번째 cross validation 학습 완료, 예측 시작')

      #학습된 lstm 저장 
      train_pred = lstm_model.predict(X_train_lstm)
      test_pred = lstm_model.predict(X_test_lstm)
      preds_dict[k] = (train_pred, test_pred)

      train_MAE = mean_absolute_error(Y_train_lstm.squeeze(), train_pred)
      test_MAE = mean_absolute_error(Y_test_lstm.squeeze(), test_pred)

      train_MAPE = mean_absolute_percentage_error(Y_train_lstm.squeeze(), train_pred)
      test_MAPE = mean_absolute_percentage_error(Y_test_lstm.squeeze(), test_pred)

      train_RMSE = mean_squared_error(Y_train_lstm.squeeze(), train_pred)
      test_RMSE = mean_squared_error(Y_test_lstm.squeeze(), test_pred)

      train_R2 = r2_score(Y_train_lstm.squeeze(), train_pred)
      test_R2 = r2_score(Y_test_lstm.squeeze(), test_pred)
      print('{}_{}_TRAIN ~ MAE, MAPE, RMSE, R2 {}, {}, {}, {} '.format(k,match_dict[k], train_MAE, train_MAPE, train_RMSE, train_R2))
      print('{}_{}_TEST ~ MAE, MAPE, RMSE, R2 {}, {}, {}, {} '.format(k,match_dict[k], test_MAE, test_MAPE, test_RMSE, test_R2))
      train_eval_dict[k] = [train_MAE, train_MAPE, train_RMSE, train_R2] # train_MAE,train_MAPE, train_RMSE, train_R2 순 
      test_eval_dict[k] = [test_MAE, test_MAPE, test_RMSE, test_R2] # test_MAE, test_MAPE, test_RMSE, test_R2 순

      train_scores.append(train_eval_dict[k])
      test_scores.append(test_eval_dict[k])
  return train_scores, test_scores 
