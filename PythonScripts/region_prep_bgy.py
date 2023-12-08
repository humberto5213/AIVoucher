#prep_bgy.py 

import pandas as pd 
import os 
import unicodedata
from dataprep.eda.missing import plot_missing
from dataprep.eda import create_report
import matplotlib.pyplot as plt 
import pickle
import re 
import numpy as np 

class PREP_BGY():
    def x_prep():
        match_info = pd.read_excel('미세먼지_간이측정망_시스템_정보.xlsx', skiprows=5)
        X_path = './국가망과 매치된 보건연데이터/'
        match_dict = dict() 
        X_dict = dict() 

        print('match info 생성 중..')
        for excel in os.listdir(X_path):
            location = excel.split('(')[-1].split(')')[0]
            location = unicodedata.normalize('NFC',location)  # 한글 자모 깨짐현상 수정 

            num = match_info[match_info['간이측정소명\n(설치지점)'].str.replace(" ","").str.contains(location) == True]['순번'].values[0].astype(int)
            match_dict[num] = location 

        with open('보건연_match_dict.pickle', 'wb') as handle:
            pickle.dump(match_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        print('match dict 생성 완료 및 pickle 저장')
        print('X dataframe 생성 중..')
        for excel in os.listdir(X_path):
            location = excel.split('(')[-1].split(')')[0]
            print(location)
            location = unicodedata.normalize('NFC',location)  # 한글 자모 깨짐현상 수정 

            num = match_info[match_info['간이측정소명\n(설치지점)'].str.replace(" ","").str.contains(location) == True]['순번'].values[0].astype(int)
            df = pd.read_excel(X_path + excel, na_values=['(P)'], usecols = "A:H")

            df_ = df[df['분'] == 0]

            time_col = df_['년'].map(str) + '-' + df_['월'].astype(int).map(str) + '-' + df_['일'].astype(int).map(str) + ':' + df_['시'].astype(int).map(str) 
            df_ =  df_.drop(columns=['년','월','일','시','분','측정소'])

            df_.insert(0, 'time', time_col)
            #plot_missing(df)
   
            print('전처리 전 Data preparation report 생성 시작..')
            report = create_report(df_, title='{}_{}_NaN처리전_보건연_정각데이터'.format(num,location))
            report.save('dataprep_보건연/{}_{}_NaN처리전_보건연_정각데이터'.format(num,location))

            df_ = df_.interpolate() #선형보간 
            # 맨 앞뒤가 nan 이면 interpolate 불가하므로 
            df_ = df_.fillna(method="ffill")
            df_ = df_.fillna(method="backfill")
         
            print('전처리 후 Data preparation report 생성 시작..')
            report = create_report(df_, title='{}_{}_NaN처리후_보건연_정각데이터'.format(num,location))
            report.save('dataprep_보건연/{}_{}_NaN처리후_보건연_정각데이터'.format(num,location))
            #print(df_.head())
            X_dict[num] = df_ 
            
        print('완성된 input X를 pickle 파일로 저장 시작..')
        with open('보건연_정각데이터_X_dict.pickle', 'wb') as handle:
            pickle.dump(X_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return match_dict, X_dict 

    def y_prep():
        # 서초구는 80 timestep 연속 NaN 도있어서 그냥 전날 데이터 복사하는게 나을수도 
        # X : 2021-11-4:00 ~ 2022-05-31:23
        # Y : 2021-11-1:01 ~ 2022-06-30:24
        Y_path = './국가망/(보건연) 국가망(21.11_22.06)/'
        Y_dict = dict() 
        print('Y dataframe 생성 중..')
        for excel in os.listdir(Y_path):
            excel = unicodedata.normalize('NFC',excel) 
            print(excel)
            nums = [int(s) for s  in re.findall(r'\b\d+\b', excel)] # 숫자만 추출 
            #print(num)
            print(nums)
            df = pd.read_excel(Y_path+excel,skiprows=5).iloc[:, [0, 2,4]]
            df.columns=['time', 'PM10', 'PM2.5']

            df = df.interpolate()
            df = df.fillna(method="ffill")
            df = df.fillna(method="backfill")

            for num in nums: 
                Y_dict[num] = df 
            
        return Y_dict 

    def datetime_indexing(X_dict, Y_dict):
        # datetime index 생성 
        for k, v in X_dict.items():
            df = v 
            df[['date', 'hour']] = df['time'].str.split(':', 1, expand=True)
            df = df.set_index(pd.to_datetime(df.date) + df.hour.astype('timedelta64[h]'))
            df = df.drop(columns=['time','date','hour'])
            X_dict[k] = df 

        for k, v in Y_dict.items():
            df = v 
            df[['date', 'hour']] = df['time'].str.split(':', 1, expand=True)
            df = df.set_index(pd.to_datetime(df.date) + df.hour.astype('timedelta64[h]'))
            df = df.drop(columns=['time','date','hour'])
            Y_dict[k] = df 

        return X_dict, Y_dict 

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