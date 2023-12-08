import pandas as pd 
import os 
import unicodedata
from dataprep.eda.missing import plot_missing
from dataprep.eda import create_report
import matplotlib.pyplot as plt 
import pickle
import re 
import numpy as np 

class PREP_KT():
    def x_prep():
        match_info = pd.read_excel('KT_matching_info.xlsx')
        X_path = './KT/'
        match_dict = dict() 
        X_dict = dict() 

        print('match info 생성 중..')
        for i in range(len(match_info)):
            match_dict[match_info.iloc[i]['INPUT']] =  match_info.iloc[i]['LABEL']

        with open('KT_match_dict.pickle', 'wb') as handle:
            pickle.dump(match_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        print('match dict 생성 완료 및 pickle 저장')
        print('X dataframe 생성 중..')
        for k in match_dict.keys():
            print(k)
            df  = pd.read_excel('./KT/KT 국소 지점 1시간단위 센서 데이터_220621.xlsx', sheet_name = k)
            df = df[['equip_msr_dt', '미세먼지', '초미세먼지']]
            df.columns = ['time', 'PM10', 'PM2.5']
            df[['date', 'hour']] = df['time'].str.split(' ', 1, expand=True)
            df.hour = df.hour.str[:2]
            df = df.set_index(pd.to_datetime(df.date) + df.hour.astype('timedelta64[h]'))
            df = df.drop(columns=['time','date','hour'])

            location = k.split('.')[-1]

            print('전처리 전 Data preparation report 생성 시작..')
            report = create_report(df, title='{}_NaN처리전_KT_정각데이터'.format(location))
            report.save('dataprep_KT/{}_NaN처리전_KT_정각데이터'.format(location))

            df = df.interpolate() #선형보간 
            # 맨 앞뒤가 nan 이면 interpolate 불가하므로 
            df = df.fillna(method="ffill")
            df = df.fillna(method="backfill")
          
            print('전처리 후 Data preparation report 생성 시작..')
            report = create_report(df, title='{}_NaN처리후_KT_정각데이터'.format(location))
            report.save('dataprep_KT/{}_NaN처리후_KT_정각데이터'.format(location))
            #print(df_.head())
            X_dict[k] = df
        
        print('완성된 input X를 pickle 파일로 저장 시작..')
        with open('KT_정각데이터_X_dict.pickle', 'wb') as handle:
            pickle.dump(X_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
     
        return match_dict, X_dict 

    def y_prep(match_dict):
        Y_path = './국가망/(KT) 국가망(22.01-06)/'
        Y_dict = dict() 

        print('Y dataframe 생성 중..')
        for excel in os.listdir(Y_path):
            print(excel)
            excel = unicodedata.normalize('NFC',excel) 

            df = pd.read_excel(Y_path+excel,skiprows=5).iloc[:, [0, 2,4]]
            df.columns=['time', 'PM10', 'PM2.5']
            df[['date', 'hour']] = df['time'].str.split(':', 1, expand=True)
            df = df.set_index(pd.to_datetime(df.date) + df.hour.astype('timedelta64[h]'))
            df = df.drop(columns=['time','date','hour'])

            df = df.interpolate()
            df = df.fillna(method="ffill")
            df = df.fillna(method="backfill")

        for k in match_dict.keys(): 
            Y_dict[k] = df 
        
        print('완성된 보정용 국가망 Y를 pickle 파일로 저장 시작..')
        with open('국가망_KT_Y_dict.pickle', 'wb') as handle:
            pickle.dump(Y_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return match_dict, Y_dict 

    def align_time(X_dict, Y_dict):
        # Y가 X보다 날짜범위가 넓으므로, 좁은범위에 맞춰줌
        for k, v in X_dict.items(): 
            diff_index = Y_dict[k].index.difference(X_dict[k].index)
            df =  Y_dict[k]
            df = df.drop(labels=diff_index)
            Y_dict[k] = df 

            diff_index = X_dict[k].index.difference(Y_dict[k].index)
            df =  X_dict[k]
            df = df.drop(labels=diff_index)
            X_dict[k] = df 

        return X_dict, Y_dict