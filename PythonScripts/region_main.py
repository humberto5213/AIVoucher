# region_main.py - cross_val mode 실험 
import pandas as pd 
import os 
import unicodedata
from dataprep.eda.missing import plot_missing
from dataprep.eda import create_report
import matplotlib.pyplot as plt 
import pickle
import re 
import numpy as np 
import argparse

from region_prep_bgy import PREP_BGY
from region_prep_KT import PREP_KT
from region_model import train, evaluate, cross_validate

# org : '보건연', 'KT' >> 보건연 데이터인지, KT 데이터 인지 선택 
# mode : 'train', 'test' >> 학습할지, 학습된 weight로 test를 할지 여부 
# load_prepared : True, False >> True 이면 input X 및 Y 에 대한 dict를 최초 1회 생성 후, pickle파일로 저장된 dict를 재활용하기 위해 load만 하고 새로 생성은 안함 
# prep_report : True, False >> NaN 값 등에 대한 데이터 전처리 전, 후에 대한 요약 레포트 저장 여부
def clipping(_X, _Y):
  for k, v in _X.items():
    pm10_max = _Y[k]['PM10'].max()
    pm2_5_max = _Y[k]['PM2.5'].max() 
    
    v.loc[v['PM10'] > pm10_max, 'PM10'] = pm10_max
    v.loc[v['PM2.5'] > pm2_5_max, 'PM2.5'] = pm2_5_max
    _X[k] = v 
  return _X, _Y 

if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument("--org", dest="org", action="store") 
  parser.add_argument("--mode", dest="mode", action="store")
  parser.add_argument("--select_region", dest="select_region", default=False, action="store")
  parser.add_argument("--load_prepared", dest="load_prepared", action="store") 
  parser.add_argument("--train_split", dest="train_split", action="store") 
  parser.add_argument("--n_split", dest="n_split", action="store") 
  args = parser.parse_args()
  '''
  ############
  org = '보건연'
  load_prepared = 'True'
  train_split = 0.7 
  mode = 'cross_val' 
  select_region = 113 ## 
  ############ 
  '''
  print(args.org, '데이터 준비')
  if args.org == '보건연':
    prep = PREP_BGY
    if args.load_prepared == 'False':
      print('X input과 보정용 국가망 Y를 처음부터 준비 시작..')
      match_dict, X_dict = prep.x_prep()
      Y_dict = prep.y_prep()
    else : 
      print('이미 만들어진 X input과 보정용 국가망 Y를 load..')
      with open('보건연_정각데이터_X_dict.pickle', 'rb') as handle:
        X_dict = pickle.load(handle)
      with open('국가망_보건연_Y_dict.pickle', 'rb') as handle:
        Y_dict = pickle.load(handle) 
    
    # 지역 한 개에 대해서만 진행 
    if args.select_region:
      select_region = int(args.select_region)
      tmp = dict() 
      tmp[select_region] = X_dict[select_region]
      X_dict = tmp 
      tmp2 = dict()
      tmp2[select_region] = Y_dict[select_region]
      Y_dict = tmp2 
    
    X_dict, Y_dict = clipping(X_dict, Y_dict) ### 
    X_dict, Y_dict = prep.datetime_indexing(X_dict, Y_dict)
    # Y가 X보다 날짜범위가 넓으므로, 좁은범위에 맞춰줌
    X_dict, Y_dict = prep.align_time(X_dict, Y_dict)
  
  elif args.org == 'KT':
    prep = PREP_KT
    if args.load_prepared == 'False':
      print('X input과 보정용 국가망 Y를 처음부터 준비 시작..')
      match_dict, X_dict = prep.x_prep()
      Y_dict = prep.y_prep(match_dict)
    else : 
      print('이미 만들어진 X input과 보정용 국가망 Y를 load..')
      with open('KT_정각데이터_X_dict.pickle', 'rb') as handle:
        X_dict = pickle.load(handle)
      with open('국가망_KT_Y_dict.pickle', 'rb') as handle:
        Y_dict = pickle.load(handle)

    # 지역 한 개에 대해서만 진행 
    if args.select_region:
      select_region = int(args.select_region)
      tmp = dict() 
      tmp[select_region] = X_dict[select_region]
      X_dict = tmp 
      tmp2 = dict()
      tmp2[select_region] = Y_dict[select_region]
      Y_dict = tmp2 
    
    X_dict, Y_dict = clipping(X_dict, Y_dict) ### 
    # Y가 X보다 날짜범위가 넓으므로, 좁은범위에 맞춰줌
    X_dict, Y_dict = prep.align_time(X_dict, Y_dict)  

  ##############################Hyper Parameter##############################
  # 원본 데이터는 1시간 간격 데이터임. 
  # >> 이전 12시간을 보고 다음 1시간의 데이터 예측하는 의미로 진행
  # 원래는 X에 대해서만 fitting 시키는 게 일반적이지만, 여기서는 Y(국가망 데이터)로 보정하기 위해 pred-Y의 Y는  X의 미래 1시점이 아니라 Y의 미래 1시점으로 대체  
  #train test는 30%로 split 
  # 결국 X_train을 학습할 때 Y_train으로 보정하고, X_test를 input으로 학습된 모델에 넣어주면 Y_test의 미래 1시점을 맞추는 것. 
  # 즉 scaler는 X_train것을 X_test에 적용하고 Y_train것을 Y_test에 적용 (Y_train도 학습 때 이미 주어진 정보라고 할 수 있으니까)
  ##############################Hyper Parameter##############################
  window_size = 12 
  lr = 0.001 #learning rate
  ep = 300 #epochs
  bs = 32 #batch size
  train_split_ratio = float(args.train_split) # e.g. train : test = 70% : 30% split 
  n_split = int(args.n_split) # cross validation 을 위한 time series 쪼개는 단위 
  selected_features = ['PM2.5','PM10'] 
  ##############################Hyper Parameter##############################

  if args.mode == 'cross_val':
    train_scores, test_scores = cross_validate(X_dict, Y_dict, args.org, selected_features, window_size, lr, ep, bs, n_split)
    print('------------------------------Cross validation 종료. 최종 Scores------------------------------')
    for idx, s in enumerate(train_scores):
      print(idx,'th cross val train - scores:', s)
    for idx, s in enumerate(test_scores):
      print(idx,'th cross val validation(test) - scores:', s)

    print('Average cross-validated Train scores [MAE, MAPE, RMSE, R2]: ', np.mean(train_scores, axis=0))
    print('Average cross-validated Test scores [MAE, MAPE, RMSE, R2]: ', np.mean(test_scores, axis=0))

  elif args.mode == 'train':
    print('train:test split ratio:', train_split_ratio,':', round(1-train_split_ratio,1))
    print('LSTM 학습 시작..')
    train(X_dict, Y_dict, args.org, selected_features, window_size, lr, ep, bs, train_split_ratio)

  elif args.mode == 'evaluate':
    print('train:test split ratio:', train_split_ratio,':', round(1-train_split_ratio,1))
    print('평가 시작..')
    evaluate(X_dict, Y_dict, args.org, selected_features, window_size, lr, ep, bs, train_split_ratio)
  