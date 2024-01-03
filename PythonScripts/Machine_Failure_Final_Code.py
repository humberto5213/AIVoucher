from easydict import EasyDict as edict
from pathlib import Path
import yaml
import pickle
from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.manifold import TSNE
import os
import random
from tqdm import tqdm
import tensorflow as tf
from random import shuffle
import glob
from copy import deepcopy
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, History
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.utils import shuffle
from plot_keras_history import plot_history
from sklearn.metrics import confusion_matrix, f1_score

#Config----------------------------------------------------------------------------------------------------------------------------
#All data is gathered  by this func
def get_config(config_filename):

    with open(config_filename, 'r',encoding='UTF8') as f:
        config_enkispathdir = f.readline().strip()
        config_datapathdir = f.readline().strip()
        config_resultpathdir = f.readline().strip()
        config_modelpathdir = f.readline().strip()
    
    config_enkispathdir = config_enkispathdir.split('=')[1].strip()
    config_datapathdir = config_datapathdir.split('=')[1].strip()
    config_resultpathdir = config_resultpathdir.split('=')[1].strip()
    config_modelpathdir = config_modelpathdir.split('=')[1].strip()
        
    conf = edict()
    
    conf.enkis_path = Path(config_enkispathdir)
    conf.data_path = Path(config_datapathdir)
    conf.result_data_path = Path(config_resultpathdir)
    conf.model_path = Path(config_modelpathdir)
    
    conf.npy_data_path = Path(conf.data_path/'npy')
    conf.pickle_data_path = Path(conf.data_path/'pickles')
    conf.original_data_path = Path(conf.data_path/'original_data')
    conf.update_data_path = Path(conf.data_path/'update_data')
    conf.original_site_data_pickle = 'location_dict.pickle'
    conf.all_site_data_pickle = 'update_location_dict.pickle'

    conf.pre_pickle = conf.pickle_data_path/'data_pump_pres_dtonic01.pickle'
    conf.pua_pickle = conf.pickle_data_path/'data_pump_acur_dtonic01.pickle'
    conf.sea_pickle = conf.pickle_data_path/'data_sensor_acur_dtonic01.pickle'

    conf.pre_label_pickle = conf.pickle_data_path/'data_label_pre.pickle'
    conf.pua_label_pickle = conf.pickle_data_path/'data_label_pua.pickle'
    conf.sea_label_pickle = conf.pickle_data_path/'data_label_sea.pickle'
    conf.eventlist_path = conf.data_path/'장애리스트.csv'
    
    conf.device_report_path = conf.result_data_path/'DEVICE'
    conf.pump_report_path = conf.device_report_path/'PUMP'
    conf.sensor_report_path = conf.device_report_path/'SENSOR'
    conf.plot_result_path = conf.result_data_path/'plots'
    
    conf.pvalue = [12, 24]
    conf.temp_value = [20, 40, -15, -20]
    conf.hum_value = [25, 40, -15, -25]
    conf.pre_value = [20, 50, 1535.0, 1590.0]
    conf.pua_value = [0, 30, 1200.0, 2950.0]
    conf.sea_value = [8845, 10361]
    
    conf.pm25_max = 300.0 
    conf.pm25_min = -250.0
    conf.pm10_max = 2400.0
    conf.pm10_min = -2000.0
    conf.pm_dataset_path = conf.pickle_data_path/'pm_dataset.pickle'
    conf.pm_TSNE_path = conf.pickle_data_path/'pm_tsne_2d.pickle'
    
    conf.sensor_model_path = conf.model_path/'dtonic.hdf5'
    conf.pm_model_path = conf.model_path/'pm_dnn.hdf5'

    return conf

def edict2dict(edict_obj):
    dict_obj = {}
    for key, vals in edict_obj.items():
        if isinstance(vals, edict):
            dict_obj[key] = edict2dict(vals)
        else:
            dict_obj[key] = vals
    return dict_obj

#DataReader----------------------------------------------------------------------------------------------------------------------------
#Used for reading the data into the modules
def load_site_data(conf):

    all_data_path = os.path.join(conf.pickle_data_path, conf.all_site_data_pickle)
    ## data load
    with open(all_data_path, 'rb') as fptr:
        site_data_dict = pickle.load(fptr)
        
    return site_data_dict
       
def get_column_dataframe(data_dict, col='', P_signal=False):

    data = pd.DataFrame()
    flag = 0

    for key in tqdm(data_dict.keys()):
        data_col = data_dict[key]

        if P_signal:
            data_col = data_col[data_col[col] == -100]
        else:
            data_col = data_col[data_col[col] != -100]

        data_col['t_index'] = data_col.time.apply(lambda _: get_time(_))
        data_col['t_index'] = pd.to_datetime(data_col['t_index'], format='%Y-%m-%d-%H-%M')
        data_col[key] = data_col[col]
        data_col = data_col.set_index('t_index')

        if flag == 0:
            flag = 1
            data = data_col[key]
        else:
            data = pd.concat([data, data_col[key]], axis=1)

    data_med = data.resample('1D').mean()
    med = data_med.median(axis=1)

    return data, med    
    
def get_p_signal(site_dict, conf, date_from, date_to):
    # col is in ['PM2.5', 'PM10', 'TEMP', 'HUM', 'PRE', 'PUA', 'SAE']      
    P_signal_df, _ = get_column_dataframe(site_dict, col ='TEMP', P_signal=True) # col name is not important
    #limit time periods
    P_signal = set_time_period(P_signal_df, date_from, date_to)
    return P_signal
    
def get_temperature_data(site_dict, conf, date_from, date_to):
    # col is in ['PM2.5', 'PM10', 'TEMP', 'HUM', 'PRE', 'PUA', 'SAE']    
    temp_df, temp_med = get_column_dataframe(site_dict, col ='TEMP', P_signal=False) 
    temp_df = set_time_period(temp_df, date_from, date_to)
    temp_med = set_time_period(temp_med, date_from, date_to)
    save_data_pickle(conf, temp_df, 'Temp', 'dtonic01')
    save_data_pickle(conf, temp_med, 'Temp_median', 'dtonic01')    
    return (temp_df, temp_med)
    
def get_humidity_data(site_dict, conf, date_from, date_to):
    # col is in ['PM2.5', 'PM10', 'TEMP', 'HUM', 'PRE', 'PUA', 'SAE']    
    hum_df, hum_med = get_column_dataframe(site_dict, col ='HUM', P_signal=False) 
    hum_df = set_time_period(hum_df, date_from, date_to)
    hum_med = set_time_period(hum_med, date_from, date_to)   
    save_data_pickle(conf, hum_df, 'Humid', 'dtonic01')
    save_data_pickle(conf, hum_med, 'Humid_median', 'dtonic01')
    return (hum_df, hum_med)
    
def get_pump_pressure_data(site_dict, conf, date_from, date_to):
    # col is in ['PM2.5', 'PM10', 'TEMP', 'HUM', 'PRE', 'PUA', 'SAE']    
    pre_df, pre_med = get_column_dataframe(site_dict, col ='PRE', P_signal=False) # Pump Pressure 

    if (date_from == "none") and (date_to == "none"):
        save_data_pickle(conf, pre_df, 'pump_pres', 'dtonic01')
        save_data_pickle(conf, pre_med, 'pump_pres_median', 'dtonic01')
        return (pre_df, pre_med) 
    else:
        pre_df = set_time_period(pre_df, date_from, date_to)
        pre_med = set_time_period(pre_med, date_from, date_to)
        save_data_pickle(conf, pre_df, 'pump_pres', 'dtonic01')
        save_data_pickle(conf, pre_med, 'pump_pres_median', 'dtonic01')
        return (pre_df, pre_med) 

def get_pump_current_data(site_dict, conf, date_from, date_to):
    # col is in ['PM2.5', 'PM10', 'TEMP', 'HUM', 'PRE', 'PUA', 'SAE']    
    pua_df, pua_med = get_column_dataframe(site_dict, col ='PUA', P_signal=False) # Pump Current
    
    if (date_from == "none") and (date_to == "none"):
        save_data_pickle(conf, pua_df, 'pump_acur', 'dtonic01')
        save_data_pickle(conf, pua_med, 'pump_acur_median', 'dtonic01')
        return (pua_df, pua_med)
    else:
        pua_df = set_time_period(pua_df, date_from, date_to)
        pua_med = set_time_period(pua_med, date_from, date_to)
        save_data_pickle(conf, pua_df, 'pump_acur', 'dtonic01')
        save_data_pickle(conf, pua_med, 'pump_acur_median', 'dtonic01')
        return (pua_df, pua_med)

def get_sensor_current_data(site_dict, conf, date_from, date_to):
    # col is in ['PM2.5', 'PM10', 'TEMP', 'HUM', 'PRE', 'PUA', 'SAE']    
    sea_df, sea_med = get_column_dataframe(site_dict, col ='SEA', P_signal=False) # Sensor Current 
    
    if (date_from == "none") and (date_to == "none"):
        save_data_pickle(conf, sea_df, 'sensor_acur', 'dtonic01')
        save_data_pickle(conf, sea_med, 'sensor_acur_median', 'dtonic01')
        return (sea_df, sea_med)
    else:
        sea_df = set_time_period(sea_df, date_from, date_to)
        sea_med = set_time_period(sea_med, date_from, date_to)
        save_data_pickle(conf, sea_df, 'sensor_acur', 'dtonic01')
        save_data_pickle(conf, sea_med, 'sensor_acur_median', 'dtonic01')
        return (sea_df, sea_med)
    
def get_pm25_data(site_dict, conf, date_from, date_to):
    # col is in ['PM2.5', 'PM10', 'TEMP', 'HUM', 'PRE', 'PUA', 'SAE']    
    pm25_df, pm25_med = get_column_dataframe(site_dict, col ='PM2.5', P_signal=False) 
    pm25_df = set_time_period(pm25_df, date_from, date_to)
    pm25_med = set_time_period(pm25_med, date_from, date_to)
    save_data_pickle(conf, pm25_df, 'PM2.5', 'dtonic01')
    save_data_pickle(conf, pm25_med, 'PM2.5_median', 'dtonic01')    
    return (pm25_df, pm25_med)

def get_pm10_data(site_dict, conf, date_from, date_to):
    # col is in ['PM2.5', 'PM10', 'TEMP', 'HUM', 'PRE', 'PUA', 'SAE']    
    pm10_df, pm10_med = get_column_dataframe(site_dict, col ='PM10', P_signal=False) 
    pm10_df = set_time_period(pm10_df, date_from, date_to)
    pm10_med = set_time_period(pm10_med, date_from, date_to)
    save_data_pickle(conf, pm10_df, 'PM10', 'dtonic01')
    save_data_pickle(conf, pm10_med, 'PM10_median', 'dtonic01')
    return (pm10_df, pm10_med)

def get_all_data(site_dict, conf, date_from, date_to):
   
    P_signal = get_p_signal(site_dict, conf, date_from, date_to)
    temp = get_temperature_data(site_dict, conf, date_from, date_to)
    hum = get_humidity_data(site_dict, conf, date_from, date_to)
    pre = get_pump_pressure_data(site_dict, conf, date_from, date_to)
    pua = get_pump_current_data(site_dict, conf, date_from, date_to)
    sea = get_sensor_current_data(site_dict, conf, date_from, date_to)
    pm25 = get_pm25_data(site_dict, conf, date_from, date_to)
    pm10 = get_pm10_data(site_dict, conf, date_from, date_to)
    
    return P_signal, temp, hum, pre, pua, sea, pm25, pm10

def get_control_parameters():    
    with open('ai_voucher/enkis/config.txt', 'r',encoding='UTF8') as f:
        f.readline()
        p_value = f.readline().strip()
        f.readline()
        temp_value = f.readline().strip()
        f.readline()
        hum_value = f.readline().strip()
        f.readline()
        pre_value = f.readline().strip()
        f.readline()
        pua_value = f.readline().strip()
        f.readline()
        sea_value = f.readline().strip()

    p_value = list(map(float,(p_value.split(':')[1].split(','))))
    temp_value = list(map(float,(temp_value.split(':')[1].split(','))))
    hum_value = list(map(float,(hum_value.split(':')[1].split(','))))
    pre_value = list(map(float,(pre_value.split(':')[1].split(','))))
    pua_value = list(map(float,(pua_value.split(':')[1].split(','))))
    sea_value = list(map(float,(sea_value.split(':')[1].split(','))))
    
    return p_value, temp_value, hum_value, pre_value, pua_value, sea_value   

#Utils----------------------------------------------------------------------------------------------------------------------------
def get_time(time_stamp):
    temp = str(time_stamp).split(':')
    ret = str(temp[0]).replace(' ', '-') + '-' + str(temp[1])
    return ret
    
def set_time_period(data_df, date_from, date_to):
    return data_df.loc[str(date_from):str(date_to)]
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
        
def get_event_list(conf):
    issuelist_df = pd.read_csv(conf.eventlist_path, sep = '\s*,\s*', engine='python')
    events_df = issuelist_df.loc[:, ['간이측정소명', 'start', 'end', '장애유형']]
    events_df['장애유형'] = events_df['장애유형'].str.replace(' ', '')
    events_df['간이측정소명'] = events_df['간이측정소명'].str.replace('&', ',')
    
    return events_df
    
def save_data_pickle(conf, data=None, colName=None, postName=None):
    # colName is in ['PM2.5', 'PM10', 'TEMP', 'HUM', 'PRE', 'PUA', 'SAE']   
    if colName is not None and data is not None:
        save_data_path = os.path.join(conf.pickle_data_path, 'data_{}_{}.pickle'.format(colName,postName))    
        with open(save_data_path, 'wb') as fptr:
            pickle.dump(data, fptr)  
    else:
        print('Wrong command parameters')
        
def load_data_pickle(conf, colName=None, postName=None):
    # colName is in ['PM2.5', 'PM10', 'TEMP', 'HUM', 'PRE', 'PUA', 'SAE']   
    if colName is not None:
        load_data_path = os.path.join(conf.pickle_data_path, 'data_{}_{}.pickle'.format(colName,postName))
        with open(load_data_path,'rb') as fptr:
            data_df = pickle.load(fptr)
        return data_df
    else:
        print('Wrong command parameters')
        return None
        
def convert_dataframe_to_list(event_df):
    eventDict = event_df.to_dict('split')  
    return eventDict['data']

def s_daily_average(data_dict, moving_value=3, spilt_value=7):

    data = dict()

    for key in data_dict.keys():
        sample = data_dict[key]
        daily_values = sample.resample('1D')
        daily_v_indeces = (daily_values.max()+daily_values.median())/2
            
        week_data = []
        count = len(daily_v_indeces)
        for i in range(count-spilt_value):
            week_sample = daily_v_indeces.iloc[i:i+spilt_value]           
            week_data.append(week_sample)
        if week_data is not None:
            data[key] = week_data
        
    return data
    
    
#-----------------------------------------------
#Raw Data Preprocessing
#-----------------------------------------------
def RunPreproc(conf):
    # data (excel) 경로
    file_list = glob.glob(str(conf.original_data_path)+'/*.xlsx')
    
    # data 전처리
    def data_set(file_list):

        data = dict()
        for file_name in tqdm(file_list):

            # (excel) file open
            df  = pd.read_excel(file_name, engine='openpyxl')
            # 측정장소명 key에 저장
            key = df['측정소'].unique()[0]
            # 하단의 data 통계 수치 제거
            df  = df.iloc[:-3]
            # 시간 정보의 date, time 컬럼 생성 및, 미수신 신호 -100 변환
            df = df.astype({'년': 'int', '월': 'int', '일': 'int', '시': 'int', '분': 'int'})
            df = df.replace({'(P)': -100})

            df['date'] = df['년'].map(str) + '-' + df['월'].map(str) + '-' + df['일'].map(str)
            df['time'] = df['년'].map(str) + '-' + df['월'].map(str) + '-' + df['일'].map(str) + ' ' + df[
                '시'].map(str) + ':' + df['분'].map(str)
            df = df.reset_index(drop=True)
            df.drop(['년', '월', '일', '시', '분', '측정소'], axis=1, inplace=True)
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M')

            # 최종 output : 장소를 key, 센서 데이터를 value로 가지는 dict 파일
            data[key] = df
        return data

    location_dict = data_set(file_list)

    data_name = os.path.join(conf.pickle_data_path, conf.original_site_data_pickle)
    # pickle 파일 생성
    with open(data_name,'wb') as f_1:
        pickle.dump(location_dict, f_1)


#-----------------------------------------
def RunPreprocWithPrevDict(conf):

    # 추가 data (excel) 경로
    file_list = glob.glob(str(conf.update_data_path)+'/*.xlsx')

    # 이전 data, dict.pickle file open
    prev_dict_name = os.path.join(conf.pickle_data_path, conf.original_site_data_pickle)
    with open(prev_dict_name, 'rb') as f_1:
        location_dict = pickle.load(f_1)

    # data 전처리
    def data_update(file_list,location_dict):
        data = dict()
        for file_name in tqdm(file_list):
            print("[RunPreprocWithPrevDict] " + file_name)
            # (excel) file open
            df = pd.read_excel(file_name, engine='openpyxl')
            # 추가된 column 삭제
            # df = df.drop(['S05', 'S0', 'L90', 'CO2', 'VIX', 'VIY', 'VIZ'], axis=1)

            # 하단의 data 통계 수치 제거
            df = df.iloc[:-3]
            # 측정장소명 key에 저장
            key = df['측정소'].unique()[0]
            # 시간 정보의 date, time 컬럼 생성 및, 미수신 신호 -100 변환
            df = df.astype({'년': 'int', '월': 'int', '일': 'int', '시': 'int', '분': 'int'})
            df = df.replace({'(P)': -100})

            df['date'] = df['년'].map(str) + '-' + df['월'].map(str) + '-' + df['일'].map(str)
            df['time'] = df['년'].map(str) + '-' + df['월'].map(str) + '-' + df['일'].map(str) + ' ' + \
                              df['시'].map(str) + ':' + df['분'].map(str)
            df = df.reset_index(drop=True)
            df.drop(['년', '월', '일', '시', '분', '측정소'], axis=1, inplace=True)
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M')

            # 최종 output : 장소를 key, 센서 데이터를 value로 가지는 dict 파일
            if key in location_dict.keys():
                data[key] = pd.concat([location_dict[key],df],axis= 0)
                data[key] = data[key].reset_index(drop=True)
            else :
                data[key] = df

        return data

    update_location_dict = data_update(file_list,location_dict)

    # 업데이트된 pickle 파일 생성
    update_data_name = os.path.join(conf.pickle_data_path, conf.all_site_data_pickle)
    with open(update_data_name,'wb') as f:
        pickle.dump(update_location_dict,f)


#-----------------------------------------------
#	Generate Dataset
#-----------------------------------------------

def data_set_pump(data_dict, value_1,value_2,value_3,value_4):
    data = dict()

    for key in tqdm(data_dict.keys()):
        #print(key)
        window_data = []
        sample = data_dict[key]
        sample = sample[sample.isnull() == False]

        High = sample.resample('1H').max()
        Low = sample.resample('1H').min()
        data_x = pd.concat([High,Low],axis=1)
        count = len(High)
        spilt_value = 8
        spilt_count = int(count/spilt_value)
        #print(spilt_count)

        for i in range(spilt_count):
            window_data.append(data_x.iloc[i*spilt_value:(i+1)*spilt_value])

        # for i in range(count-spilt_value):
        #     window_data.append(data_x.iloc[i:i+spilt_value])

        label_data = []
        for window_idx, window in enumerate(window_data):
            label_window = []
            for idx, element in enumerate(window.iloc):
                label_value = []
                for ele_idx, i in enumerate(element):
                    value = 0
                    if ((i >= value_3) & (i < value_4)) | ((i > value_1) & (i <= value_2)):
                        value = 1
                    # alarm
                    if (i <= value_1) | (i >= value_4):
                        value = 2
                    if np.isnan(i) == True:
                        break
                    label_value.append(value)
                    
                if label_value :
                    high = label_value[0]
                    low = label_value[1]
                    if (high == 2) | (low == 2):
                        high = 2
                    if ((high != 2) | (low != 2)) & ((high == 1) | (low == 1)):
                        high = 1
                    label_window.append(high)
            if len(label_window) == 8:
                label_data.append(label_window)

        data[key] = label_data
    return data

def data_set_sea(data_dict, value_1,value_2):
    data = dict()
    for key in tqdm(data_dict.keys()):
        #print(key)
        window_data = []
        sample = data_dict[key]
        sample = sample[sample.isnull() == False]
        label_data = []

        High = sample.resample('1H').max()
        count = len(High)
        spilt_value = 8
        for i in range(count-spilt_value):
            window_data.append(High.iloc[i:i+spilt_value])

        for window_idx, window in enumerate(window_data):
            label_window = []
            for idx, element in enumerate(window):
                value = 0
                # warning
                if ((element >= value_1) & (element < value_2)):
                    value = 1
                # alarm
                if (element >= value_2) :
                    value = 2
                if np.isnan(element) == True:
                    break
                label_window.append(value)
            if len(label_window) == 8:
                label_data.append(label_window)
        data[key] = label_data

    return data

def data_label(data_dict):

    label_dict = dict()
    for key in tqdm(data_dict.keys()):
        #print(key)
        sample = data_dict[key]
        label = np.zeros(len(sample), dtype=int)
        for idx, window in enumerate(sample):
            sum = np.mean(window)
            if sum < 0.7:
                label[idx] = 0
            elif (sum < 1.6):
                label[idx] = 1
            else :
                label[idx] = 2

        label_dict[key] = [sample, label]

    return label_dict


def generate_data_set(conf):
	with open(conf.pre_pickle, 'rb') as f_1:
		data_pre = pickle.load(f_1)

	with open(conf.pua_pickle, 'rb') as f_1:
		data_pua = pickle.load(f_1)

	with open(conf.sea_pickle, 'rb') as f_1:
		data_sea = pickle.load(f_1)

	data_window_pre = data_set_pump(data_pre,1335,1467,1550,1625)
	data_window_pua = data_set_pump(data_pua,450,566,1872,2232)
	data_window_sea = data_set_sea(data_sea,5845,11036)

	data_label_pre = data_label(data_window_pre)
	data_label_pua = data_label(data_window_pua)
	data_label_sea = data_label(data_window_sea)   

	with open(conf.pre_label_pickle, 'wb') as f_1:
		pickle.dump(data_label_pre,f_1)

	with open(conf.pua_label_pickle, 'wb') as f_1:
		pickle.dump(data_label_pua,f_1)

	with open(conf.sea_label_pickle, 'wb') as f_1:
		pickle.dump(data_label_sea,f_1)

	return data_label_pre, data_label_pua, data_label_sea
    
#CheckP-Signals----------------------------------------------------------------------------------------------------------------------------
matplotlib.rcParams['axes.unicode_minus'] =False
    
def check_P_signals(p_df, col, conf, date_from=None, date_to=None, drawOp=False):
    
    if date_from == None:
        date_from = p_df.iloc[0].name 
        date_to = p_df.iloc[-1].name
        
    p_value = conf.pvalue

    for key in p_df.keys():

        p_count = 0
        warning_start_date = []
        warning_end_date = []
        alarm_start_date = []
        alarm_end_date = []
        temp_date = []
        p_warn = pd.DataFrame()
        p_alarm = pd.DataFrame()

        result_path = './dtonic_result/P_signal/'+key
        sample = p_df[key].loc[str(date_from):str(date_to)]
        sample = sample.resample('1H').sum().apply(lambda x: x * -1 / 100 if x < 0 else x)
           
        for idx, value in zip(sample.index, sample):

            if value == p_value[0]:
                temp_date.append(idx)
                p_count += 1

            if value != p_value[0]:
                if p_count >= p_value[0] and p_count < p_value[1]:
                    warning_start_date.append(str(temp_date[-p_count]))
                    warning_end_date.append(str(temp_date[-1]))

                elif p_count >= p_value[1]:
                    alarm_start_date.append(str(temp_date[-p_count]))
                    alarm_end_date.append(str(temp_date[-1]))

                p_count = 0
                temp_date = []

        for start,end in zip(warning_start_date,warning_end_date):
            temp = sample[(sample.index >= start) & (sample.index <= end)]
            p_warn = pd.concat([p_warn, temp], axis=0)

        for start,end in zip(alarm_start_date,alarm_end_date):
            temp = sample[(sample.index >= start) & (sample.index <= end)]
            p_alarm = pd.concat([p_alarm, temp], axis=0)

        month_start_date = []
        month_end_date = []
        
        for idx, month_start in zip(sample.index, sample.index.is_month_start):
            if month_start == True:
                month_start_date.append((idx.date()))

        for idx, month_end in zip(sample.index, sample.index.is_month_end):
            if month_end == True:
                month_end_date.append((idx.date()))


        if (len(p_warn) > 0) | (len(p_alarm)) > 0:

            result_path = Path(conf.result_data_path/'P_signal/')
            result_path = os.path.join(result_path, key)
            print(result_path)
            createFolder(result_path)

            if drawOp:
                plt.ioff()
                for start_day, end_day in zip(sorted(set(month_start_date)), sorted(set(month_end_date))):
                    year, month, _ = str(start_day).split('-')
                    fig, ax = plt.subplots(figsize= (20,10))
                    ax.plot(sample.index, sample, alpha=0.2, color='blue')
                    ax.scatter(p_alarm.index, p_alarm, color='red')
                    ax.scatter(p_warn.index, p_warn, color='blue')
                    ax.set_xlim([start_day, end_day])
                    fig.autofmt_xdate()
                    plt.title("P_signal : " + month + "월")
                    plt.savefig(result_path + '/' + key + '_' + year +'년'+month+'월'+'.png')
                plt.close()

            file = open(result_path+'/'+key + "_P_signal.txt", "w")
            file.write(" 점검지역 : " + key + '\n')
            file.write(" 점검기간 : {} ~ {}\n".format(str(date_from.date()), str(date_to.date())))
            file.write(" 점검사항 : 미수신 이벤트 [주의] {} 번, [경고] {} 번\n".format(len(warning_start_date), len(alarm_start_date)))
            file.write('\n 미수신 발생 주의 기간 : \n')
            for start,end in zip(warning_start_date,warning_end_date):
                file.write(" 시작 : " + str(start)+' , '+" 끝 :"+str(end)+'\n')
            file.write('\n 미수신 발생 경고 기간 : \n')
            for start,end in zip(alarm_start_date,alarm_end_date):
                file.write(" 시작 : " + str(start)+' , '+" 끝 :"+str(end)+'\n')
            file.close()

#HumidityAnalysis----------------------------------------------------------------------------------------------------------------------------
matplotlib.rcParams['axes.unicode_minus'] =False

def humidity_analysis(hum_set, conf, date_from=None, date_to=None, drawOp=False):

    hum_df, hum_med = hum_set
    if date_from == None:
        date_from = hum_df.iloc[0].name 
        date_to = hum_df.iloc[-1].name
        
    hum_value = conf.hum_value

    for key in hum_df.columns:

        warn_date = []
        alarm_date = []
        sample = hum_df[key].resample('1D').mean()
        sample = sample.loc[str(date_from):str(date_to)]

        x = sample - hum_med

        check = x[(x >= hum_value[0]) | (x <= hum_value[2])]

        if (len(check) > 0) :
            result_path = Path(conf.result_data_path/'HUMIDITY/')
            result_path = os.path.join(result_path, key)
            print(result_path)
            createFolder(result_path)            

            for idx, value in zip(x.index,x):
                value = round(value,2)
                if (value >= hum_value[0] and value <= hum_value[1]) | (value > hum_value[3] and value <= hum_value[2]):
                    warn_date.append(idx.date())
                if (value > hum_value[1]) | (value <= hum_value[3]):
                    alarm_date.append(idx.date())
            hum_warn = sample[warn_date]
            hum_alarm = sample[alarm_date]
            
            if drawOp:
                fig, ax = plt.subplots(figsize= (20,10))
                ax.plot(sample, alpha=0.2,color='black', label ='data')
                ax.plot(hum_med, alpha=0.8, color='black',label='median')
                ax.scatter(hum_warn.index,hum_warn,color='blue', label ='warn')
                ax.scatter(hum_alarm.index,hum_alarm,color='red', label ='alarm')
                ax.set_xlim([date_from.date(), date_to.date()])
                fig.autofmt_xdate()

                plt.title("Humidity")
                plt.legend()
                plt.savefig(result_path + '/' + key + '_hum.png')
                plt.close()

            with open(result_path + '/' + key + "_hum.txt", "w") as file:
                file.write(" 점검지역 : " + key + '\n')
                file.write(" 점검기간 : {} ~ {}\n".format(str(date_from.date()), str(date_to.date())))
                file.write(" 점검사항 : 습도센서 이상 이벤트 [주의] {} 번, [경고] {} 번\n".format(len(hum_warn), len(hum_alarm)))
                file.write('\n 습도센서 이상징후 주의 기간 :')
                for date in warn_date:
                    file.write('\n' + str(date))
                file.write('\n\n 습도센서 이상징후 경고 기간 : ')
                for date in alarm_date:
                    file.write('\n' + str(date))
                    
#TempAnalysis----------------------------------------------------------------------------------------------------------------------------
matplotlib.rcParams['axes.unicode_minus'] =False

def temperature_analysis(temp_set, conf, date_from=None, date_to=None, drawOp=False):

    temp_df, temp_med = temp_set
    if date_from == None:
        date_from = temp_df.iloc[0].name 
        date_to = temp_df.iloc[-1].name
        
    temp_value = conf.temp_value

    for key in temp_df.columns:
       
        warn_date = []
        alarm_date = []
        sample = temp_df[key].resample('1D').mean()
        sample = sample.loc[str(date_from):str(date_to)]

        x = sample - temp_med

        check = x[(x >= temp_value[0]) | (x <= temp_value[2])]

        if len(check) > 0 :
            result_path = Path(conf.result_data_path/'TEMP/')
            result_path = os.path.join(result_path, key)
            print(result_path)
            createFolder(result_path)

            for idx, value in zip(x.index,x):
                value = round(value,2)
                if (value >= temp_value[0] and value <= temp_value[1]) | (value > temp_value[3] and value <= temp_value[2]):
                    warn_date.append(idx.date())
                if (value > temp_value[1]) | (value <= temp_value[3]):
                    alarm_date.append(idx.date())
            temp_warn = sample[warn_date]
            temp_alarm = sample[alarm_date]

            if drawOp:
                fig, ax = plt.subplots(figsize= (20,10))
                plt.ioff()
                ax.plot(sample, alpha=0.2,color='black', label ='data')
                ax.plot(temp_med, alpha=0.8, color='black',label='median')
                ax.scatter(temp_warn.index,temp_warn,color='blue', label ='warn')
                ax.scatter(temp_alarm.index,temp_alarm,color='red', label ='alarm')
                ax.set_xlim([date_from.date(), date_to.date()])
                fig.autofmt_xdate()

                plt.title('Temperature')
                plt.legend()
                plt.savefig(result_path + '/' + key + '_temp.png')
                plt.close()

            with open(result_path + '/' + key + "_temp.txt", "w") as file:
                file.write(" 점검지역 : " + key + '\n')
                file.write(" 점검기간 : {} ~ {}\n".format(str(date_from.date()), str(date_to.date())))
                file.write(" 점검사항 : 온도센서 이상 이벤트 [주의] {} 번, [경고] {} 번\n".format(len(temp_warn), len(temp_alarm)))
                file.write('\n 온도센서 이상징후 주의 기간 :')
                for date in warn_date:
                    file.write('\n' +str(date))
                file.write('\n\n 온도센서 이상징후 경고 기간 : ')
                for date in alarm_date:
                    file.write('\n' +str(date))
                    
#pump_analysis----------------------------------------------------------------------------------------------------------------------------
def set_label_timezone(timezone, conf):

    Pre_alarm_LB = conf.pre_value[0]
    Pre_warn_LB = conf.pre_value[1]
    Pre_warn_UB = conf.pre_value[2]
    Pre_alarm_UB = conf.pre_value[3]
    
    Pua_alarm_LB = conf.pua_value[0]
    Pua_warn_LB = conf.pua_value[1]
    Pua_warn_UB = conf.pua_value[2]
    Pua_alarm_UB = conf.pua_value[3]

    pre_label_dataset = [] # daily report only for timezone section
    pua_label_dataset = [] # daily report only for timezone section
    for idx in range(len(timezone)):
        data_df = timezone[idx]
        # Pump Presuure
        pre_high = data_df['High(PRE)'].to_numpy()
        pre_low = data_df['Low(PRE)'].to_numpy()
        label_window = np.empty([8]) 
        for i, (v_high, v_low) in enumerate(zip(pre_high, pre_low)):
            if np.isnan(v_high) or np.isnan(v_low):
                label = 3 # P-signal
            elif v_high > Pre_alarm_UB or v_low < Pre_alarm_LB:
                label = 2
            elif v_high > Pre_warn_UB or v_low < Pre_warn_LB:
                label = 1
            else:
                label = 0
            label_window[i] = int(label)             
        pre_label_dataset.append(label_window.astype(int))

        
        #Pum Current
        pua_high = data_df['High(PUA)'].to_numpy()
        pua_low = data_df['Low(PUA)'].to_numpy()        
        label_window = np.empty([8]) 
        for i, (v_high, v_low) in enumerate(zip(pua_high, pua_low)):
            if np.isnan(v_high) or np.isnan(v_low):
                label = 3 # P-signal
            elif v_high > Pua_alarm_UB or v_low < Pua_alarm_LB:
                label = 2
            elif v_high > Pua_warn_UB or v_low < Pua_warn_LB:
                label = 1
            else:
                label = 0
            label_window[i] = int(label)           
        pua_label_dataset.append(label_window.astype(int))    
        
    return pre_label_dataset, pua_label_dataset

def ai_evaluation(model, data_array):
    
    psig_idx_list, eval_idx_list = [], []
    for i, data in enumerate(data_array):
        if max(data) == 3:
            psig_idx_list.append(i)
        else:
            eval_idx_list.append(i)

    eval_arr = np.empty((0,8), dtype=float)
    for i in range(len(data_array)):
        if i in eval_idx_list:
            eval_arr = np.vstack((eval_arr, data_array[i]))
    
    label_data = np.reshape(eval_arr, (np.shape(eval_arr)[0], np.shape(eval_arr)[1], 1))
    label_data = label_data/2
    pred_Y = model.predict(label_data)
    pred_y = []
    for idx in range(len(pred_Y)):
        y = pred_Y[idx]
        y_index = np.argmax(y)
        pred_y = np.append(pred_y, y_index)
    predY = pred_y.astype(np.int32)

    ai_report, cnt = [], 0
    for idx in range(len(data_array)):
        if idx in psig_idx_list:
            result = 3 # p-signal
        else:
            result = predY[cnt]
            cnt += 1
        ai_report.append(result)
    
    return ai_report

def pump_analysis(pre_df, pua_df, model, conf, date_from, date_to):
    print("Starting Pump Analysis")
    #beto did-------------------------------------------------------------------------------------------------
    temp_list_1 = []
    #beto did-------------------------------------------------------------------------------------------------

    report_all = dict()
    message = ["None", "Warn", "Alarm"]
    Pre_alarm_LB = conf.pre_value[0]
    Pre_warn_LB = conf.pre_value[1]
    Pre_warn_UB = conf.pre_value[2]
    Pre_alarm_UB = conf.pre_value[3]
    
    Pua_alarm_LB = conf.pua_value[0]
    Pua_warn_LB = conf.pua_value[1]
    Pua_warn_UB = conf.pua_value[2]
    Pua_alarm_UB = conf.pua_value[3]

    scount = 0
    for key in pre_df.keys():
        result_path = Path(conf.result_data_path/'PUMP/')
        result_path = os.path.join(result_path, key)
        
        pre_sample = pre_df[key]
        pre_sample = pre_sample.loc[str(date_from):str(date_to)]
        
        pua_sample = pua_df[key]
        pua_sample = pua_sample.loc[str(date_from):str(date_to)]

        pre_High = pre_sample.resample('1H').max()
        pre_Low = pre_sample.resample('1H').min()
        pua_High = pua_sample.resample('1H').max()
        pua_Low = pua_sample.resample('1H').min()
        
        data_x = pd.concat([pre_High, pre_Low, pua_High, pua_Low], 
                           keys=['High(PRE)', 'Low(PRE)', 'High(PUA)', 'Low(PUA)'], axis=1)
        data_x['datetime'] = pre_High.index
        data_x['Date'] = [d.date() for d in data_x['datetime']]
        data_x['Time'] = [d.time() for d in data_x['datetime']]
        data_x = data_x.drop('datetime', axis=1)

        count = len(pre_High)
        timezone1 = [] # from 00H to 08H
        timezone2 = [] # from 08H to 16H
        timezone3 = [] # from 16H to 24H
        date_string = []
        for i in range(0, count, 24): # daily report
            timezone1.append(data_x.iloc[i    : i+8])
            timezone2.append(data_x.iloc[i+8  : i+16])
            timezone3.append(data_x.iloc[i+16 : i+24])   
            date_val = data_x['Date'].iloc[i]
            date_string.append(date_val.strftime('%Y-%m-%d'))

        pre_label_tzone1, pua_label_tzone1 = set_label_timezone(timezone1, conf)
        pre_label_tzone2, pua_label_tzone2 = set_label_timezone(timezone2, conf)
        pre_label_tzone3, pua_label_tzone3 = set_label_timezone(timezone3, conf)
        
        pre_tz1_report = ai_evaluation(model, pre_label_tzone1)
        pre_tz2_report = ai_evaluation(model, pre_label_tzone2)
        pre_tz3_report = ai_evaluation(model, pre_label_tzone3)
        
        pua_tz1_report = ai_evaluation(model, pua_label_tzone1)
        pua_tz2_report = ai_evaluation(model, pua_label_tzone2)
        pua_tz3_report = ai_evaluation(model, pua_label_tzone3)
        
        total_warns, total_alarms = 0, 0    
        site_report = []
        for idx, date_str in enumerate(date_string):
            pre_tz1 = pre_tz1_report[idx]
            pre_tz2 = pre_tz2_report[idx]
            pre_tz3 = pre_tz3_report[idx]
            pua_tz1 = pua_tz1_report[idx]
            pua_tz2 = pua_tz2_report[idx]
            pua_tz3 = pua_tz3_report[idx]
            tz1_status = max(pre_tz1, pua_tz1)
            tz2_status = max(pre_tz2, pua_tz2)
            tz3_status = max(pre_tz3, pua_tz3)
            found = False
            daily_report = ""
            if tz1_status > 0 and tz1_status < 3: 
                daily_report = "[00H~08H] : "
                if pre_tz1 > 0:
                    daily_report += "{}(펌프압력) ".format(message[pre_tz1])
                if pua_tz1 > 0:
                    daily_report += "{}(펌프전류) ".format(message[pua_tz1])
                found = True
            
            if tz2_status > 0 and tz2_status < 3: 
                if found: daily_report += ", "
                daily_report += "[08H~16H] : "
                if pre_tz2 > 0:
                    daily_report += "{}(펌프압력) ".format(message[pre_tz2])
                if pua_tz2 > 0:
                    daily_report += "{}(펌프전류) ".format(message[pua_tz2])
                found = True
                                                       
            if tz3_status > 0 and tz3_status < 3: 
                if found: daily_report += ", "
                daily_report += "[16H~24H] : "
                if pre_tz3 > 0:
                    daily_report += "{}(펌프압력) ".format(message[pre_tz3])
                if pua_tz3 > 0:
                    daily_report += "{}(펌프전류) ".format(message[pua_tz3])
                                                       
            daily_status = max(tz1_status, tz2_status, tz3_status)  
            if daily_status == 0 or daily_status == 3: continue                                                
            headline = "{} : Daily Report = {}\t{}".format(date_str, message[daily_status], daily_report)
            site_report.append(headline)                              
            if daily_status == 1:
                #beto did-------------------------------------------------------------------------------------------------
                temp_bool = (key, date_str, message[daily_status], daily_report)
                temp_list_1.append(temp_bool)
                #beto did-------------------------------------------------------------------------------------------------
                total_warns += 1
            else:
                #beto did-------------------------------------------------------------------------------------------------
                temp_bool = (key, date_str, message[daily_status], daily_report)
                temp_list_1.append(temp_bool)
                #beto did-------------------------------------------------------------------------------------------------
                total_alarms += 1
            
        # create a report file if this site has any issue(s)    
        if 0.2*total_warns + total_alarms >= 1:
            scount += 1
            createFolder(result_path)
            with open(result_path + '/' + key + "_pump.txt", "w") as file:
                file.write(" 점검지역 : " + key + '\n')
                
                print("* 점검지역 : {} >>> 점검사항 : 펌프 이상 이벤트 [주의] {} 번, [경고] {} 번".format(
                    key, total_warns, total_alarms) )
                file.write("* 점검기간 : {} ~ {}\n".format(str(date_from.date()), str(date_to.date())))
                file.write("* 점검사항 : 펌프 이상 이벤트 [주의] {} 번, [경고] {} 번\n".format(total_warns, total_alarms))
                file.write("  Pump Press Mornitoring: LB(alarm)={:.1f}, LB(warn)={:.1f}, UB(warn)={:.1f}, UB(alarm)={:.1f}\n".format( \
                    Pre_alarm_LB, Pre_warn_LB, Pre_warn_UB, Pre_alarm_UB))
                file.write("  Pump Current Mornitoring: LB(alarm)={:.1f}, LB(warn)={:.1f}, UB(warn)={:.1f}, UB(alarm)={:.1f}\n".format( \
                    Pua_alarm_LB, Pua_warn_LB, Pua_warn_UB, Pua_alarm_UB))
                file.write('\n* 펌프 이상징후 일간 보고 >>>\n')
                for headline in site_report:
                    print(headline)
                    file.write(headline+"\n")    
            
            report_all[key] = [ (total_warns, total_alarms), headline ]
        
    print('total sites =', scount)
    
    #beto did-------------------------------------------------------------------------------------------------
    final_temp_list.append(temp_list_1)
    #beto did-------------------------------------------------------------------------------------------------

    return report_all

#sensor_analysis----------------------------------------------------------------------------------------------------------------------------
def set_sensor_label_timezone(timezone, conf):

    warn_UB = conf.sea_value[0]
    alarm_UB = conf.sea_value[1]
    
    label_dataset = [] # daily report only for timezone section
    for idx in range(len(timezone)):
        data_df = timezone[idx]
        # Pump Presuure
        high_amp = data_df['High'].to_numpy()
        low_amp = data_df['Low'].to_numpy()
        label_window = np.empty([8]) 
        for i, (v_high, v_low) in enumerate(zip(high_amp, low_amp)):
            if np.isnan(v_high) or np.isnan(v_low):
                label = 3 # P-signal
            elif v_high > alarm_UB:
                label = 2
            elif v_high > warn_UB:
                label = 1
            else:
                label = 0
            label_window[i] = int(label)             
        label_dataset.append(label_window.astype(int))
                
    return label_dataset

def ai_evaluation(model, data_array):
    
    psig_idx_list, eval_idx_list = [], []
    for i, data in enumerate(data_array):
        if max(data) == 3:
            psig_idx_list.append(i)
        else:
            eval_idx_list.append(i)

    eval_arr = np.empty((0,8), dtype=float)
    for i in range(len(data_array)):
        if i in eval_idx_list:
            eval_arr = np.vstack((eval_arr, data_array[i]))

    label_data = np.reshape(eval_arr, (np.shape(eval_arr)[0], np.shape(eval_arr)[1], 1))
    label_data = label_data/2
    pred_Y = model.predict(label_data)
    pred_y = []
    for idx in range(len(pred_Y)):
        y = pred_Y[idx]
        y_index = np.argmax(y)
        pred_y = np.append(pred_y, y_index)
    predY = pred_y.astype(np.int32)

    ai_report, cnt = [], 0
    for idx in range(len(data_array)):
        if idx in psig_idx_list:
            result = 3 # p-signal
        else:
            result = predY[cnt]
            cnt += 1
        ai_report.append(result)
    
    return ai_report

def sensor_analysis(sea_df, model, conf, date_from, date_to):
    print("Starting Sensor Analysis")
    
    #beto did-------------------------------------------------------------------------------------------------
    temp_list_1 = []
    #beto did-------------------------------------------------------------------------------------------------
    
    report_all = dict()
    message = ["None", "Warn", "Alarm"]
    warn_UB = conf.sea_value[0]
    alarm_UB = conf.sea_value[1]
    scount = 0

    for key in sea_df.keys():
        result_path = Path(conf.result_data_path/'SENSOR/')
        result_path = os.path.join(result_path, key)
        
        timezone1 = np.nan # from 00H to 08H
        timezone2 = np.nan # from 08H to 16H
        timezone3 = np.nan # from 16H to 24H
        sample = sea_df[key]
        sample = sample.loc[str(date_from):str(date_to)]
        sample = sample[sample.isnull() == False]
        High = sample.resample('1H').max()
        Low = sample.resample('1H').min()
        data_x = pd.concat([High, Low], keys=['High', 'Low'], axis=1)
        data_x['datetime'] = High.index
        data_x['Date'] = [d.date() for d in data_x['datetime']]
        data_x['Time'] = [d.time() for d in data_x['datetime']]
        data_x = data_x.drop('datetime', axis=1)
            
        count = len(High)
        date_string = []
        timezone1 = [] # from 00H to 08H
        timezone2 = [] # from 08H to 16H
        timezone3 = [] # from 16H to 24H
        for i in range(0, count, 24): # daily report
            timezone1.append(data_x.iloc[i    : i+8])
            timezone2.append(data_x.iloc[i+8  : i+16])
            timezone3.append(data_x.iloc[i+16 : i+24])   
            date_val = data_x['Date'].iloc[i]
            date_string.append(date_val.strftime('%Y-%m-%d'))

        sea_label_tzone1 = set_sensor_label_timezone(timezone1, conf)
        sea_label_tzone2 = set_sensor_label_timezone(timezone2, conf)
        sea_label_tzone3 = set_sensor_label_timezone(timezone3, conf)

        tz1_report = ai_evaluation(model, sea_label_tzone1)
        tz2_report = ai_evaluation(model, sea_label_tzone2)
        tz3_report = ai_evaluation(model, sea_label_tzone3)
        
        total_warns, total_alarms = 0, 0
        alarm_list = []
        message = ["정상", "주의", "경고"]
        site_report = []
        for idx, date_str in enumerate(date_string):
            sea_tz1 = tz1_report[idx]
            sea_tz2 = tz2_report[idx]
            sea_tz3 = tz3_report[idx]
            found = False
            daily_report = ""
            if sea_tz1 > 0 and sea_tz1 < 3: 
                daily_report += "[00H~08H] : {}".format(message[sea_tz1])
                found = True
            
            if sea_tz2 > 0 and sea_tz2 < 3: 
                if found: daily_report += ", "
                daily_report += "[08H~16H] : {}".format(message[sea_tz2])
                found = True
                                                       
            if sea_tz3 > 0 and sea_tz3 < 3: 
                if found: daily_report += ", "
                daily_report += "[16H~24H] : {}".format(message[sea_tz3])
                                                       
            daily_status = max(sea_tz1, sea_tz2, sea_tz3)  
            if daily_status == 0 or daily_status == 3: continue                                                
            headline = "{} : Daily Report = (과전류){}\t{}".format(date_str, message[daily_status], daily_report)
            site_report.append(headline)                              
            if daily_status == 1:
                #beto did-------------------------------------------------------------------------------------------------
                temp_bool = (key, date_str, message[daily_status], daily_report)
                temp_list_1.append(temp_bool)
                #beto did-------------------------------------------------------------------------------------------------
                total_warns += 1
            else:
                #beto did-------------------------------------------------------------------------------------------------
                temp_bool = (key, date_str, message[daily_status], daily_report)
                temp_list_1.append(temp_bool)
                #beto did-------------------------------------------------------------------------------------------------
                total_alarms += 1
            
        # create a report file if this site has any issue(s)    
        if 0.2*total_warns + total_alarms >= 1:
            scount += 1
            createFolder(result_path)
            with open(result_path + '/' + key + "_sensor.txt", "w") as file:
                file.write(" 점검지역 : " + key + '\n')
                print(" 점검지역 : {} >>> 점검사항 : 미세먼지 센서 이상 [주의] {} 번, [경고] {} 번".format(
                    key, total_warns, total_alarms) )
                file.write(" 점검기간 : {} ~ {}\n".format(str(date_from.date()), str(date_to.date())))
                file.write(" 점검사항 : 미세먼지 센서 과전류 [주의] {} 번, [경고] {} 번\n".format(total_warns, total_alarms))
                file.write(" 센서 과전류 감시 : UB(주의)={:.1f}, UB(경고)={:.1f}\n".format(warn_UB, alarm_UB))
                file.write('\n 미세먼지 센서 이상징후 일일 보고 >>>\n')
                for headline in site_report:
                    print(headline)
                    file.write(headline+"\n")      
                    
            report_all[key] = [ (total_warns, total_alarms), site_report ]
        
    print('total sites =', scount)
    
    #beto did-------------------------------------------------------------------------------------------------
    final_temp_list.append(temp_list_1)
    #beto did-------------------------------------------------------------------------------------------------
    
    return report_all
    
def RunCheckDevices(conf, date_from, date_to, output_filename):
    #beto did---------------------------------
    #final_temp_list = []
    #beto did---------------------------------

    events = get_event_list(conf)
    event_list = convert_dataframe_to_list(events)
    abnormal_list = [ event[0] for event in event_list ]

    pre_df = load_data_pickle(conf, 'pump_pres', 'dtonic01')
    pua_df = load_data_pickle(conf, 'pump_acur', 'dtonic01')
    sea_df = load_data_pickle(conf, 'sensor_acur', 'dtonic01')

    pre_df = set_time_period(pre_df, date_from, date_to)
    pua_df = set_time_period(pua_df, date_from, date_to)
    sea_df = set_time_period(sea_df, date_from, date_to)

    model = load_model(conf.sensor_model_path)

    os.system('cls')
    print("\n\n미세먼지 센서 이상 분석\n")
    sensor_report = sensor_analysis(sea_df, model, conf, date_from, date_to)
    print("\n펌프 장치 이상 분석\n")
    pump_report = pump_analysis(pre_df, pua_df, model, conf, date_from, date_to)

    true_negative_list = []
    false_negative_list = deepcopy(abnormal_list)
    false_positive_list = []
    false_negative_list

    for key, _ in sensor_report.items():
        if key in abnormal_list:
            true_negative_list.append(key)
            for index, site in enumerate(false_negative_list):
                if site == key:
                    false_negative_list.pop(index)
                    break
        else:
            false_positive_list.append(key)

    for key, _ in pump_report.items():
        if key in abnormal_list:
            if key not in true_negative_list:
                true_negative_list.append(key)
            if key in false_negative_list:
                for index, site in enumerate(false_negative_list):
                    if site == key:
                        false_negative_list.pop(index)
                        break
            if key in false_positive_list:
                for index, site in enumerate(false_positive_list):
                    if site == key:
                        false_positive_list.pop(index)
                        break            
        else:
            false_positive_list.append(key)

    N = len(sea_df.keys())
    TN = len(true_negative_list)
    FP = len(false_positive_list)
    FN = len(false_negative_list)
    TP = N - TN - FP - FN

    with open("true_negatives.txt", "w") as f1:
        str_report = '\n'.join(map(str,true_negative_list))
        f1.write(str_report)
        f1.write("\n총 {}개".format(TN))

    with open("false_negatives.txt", "w") as f2:
        str_report = '\n'.join(map(str,false_negative_list))
        f2.write(str_report)
        f2.write("\n총 {}개".format(FN))

    with open("false_positives.txt", "w") as f3:
        str_report = '\n'.join(map(str,false_positive_list))
        f3.write(str_report)
        f3.write("\n총 {}개".format(FP))

    Accuracy = (TP+TN)/N
    print(TP)
    print(TN)
    
    Precision_normal = TP / (TP + FN)
    Recall_normal = TP / (TP + FP)
    Precision_abnormal = TN / (FP + TN)
    Recall_abnormal = TN / (FN + TN)    
    precision = (Precision_normal+Precision_abnormal)/2
    recall = (Recall_normal+Recall_abnormal)/2
    f1_score = 2*precision*recall / (precision + recall)

    print("\noverall_report\n")
    with open("overall_report.txt", "w") as f4:
        f4.write("TP={}, TN={}, FP={}, FN={}, All={}\n".format(TP,TN,FP,FN,N))
        print("TP={}, TN={}, FP={}, FN={}, All={}".format(TP,TN,FP,FN,N))
        f4.write("Accuracy={:.4f}, Precisio={:.4f}, Recall={:.4f}\n".format(Accuracy, precision, recall))
        print("Accuracy={:.4f}, Precisio={:.4f}, Recall={:.4f}".format(Accuracy, precision, recall))
        f4.write("F1-scorey={:.4f}\n".format(f1_score))
        print("F1-scorey={:.4f}".format(f1_score))

    final_df = pd.DataFrame(columns=['Area','Date','Message','Time'])

    for i in final_temp_list:
        for y in tqdm(i):
            final_df.loc[len(final_df)] = list(y)

    for num in tqdm(range(len(final_df))):
        a = final_df['Time'].iloc[num]
        temp_dict = dict()
        for i in a.split(','):
            temp_dict[i.split(':')[0].strip()] = i.split(':')[1].strip()
            for key in list(temp_dict.keys()):
                final_df.at[num, key] = temp_dict[key]
    final_df[['00H~08H', '08H~16H','16H~24H']] = final_df[['[00H~08H]','[08H~16H]','[16H~24H]']]
    final_df = final_df[['Area', 'Date', 'Message','00H~08H', '08H~16H', '16H~24H']]
    final_df.fillna(value='없음', inplace=True)
    final_df.Message = final_df.Message.map({'Warn':'주의', 'Alarm':'경고', '경고':'경고', '주의':'주의'})      
        
    final_df.to_csv(output_filename)
    print("report_file_loc: {}".format(output_filename))

####################################################################################
############################Run All Modules for Analysis############################
####################################################################################

def RunAllModules(date_from, date_to, output_filename, config_filename):
    yaml.dump(edict2dict(get_config(config_filename)), open('config.yaml', 'w'), default_flow_style=False)

    # For Turning off the warning from pandas
    pd.set_option('mode.chained_assignment',  None)

    print('Dtonic Solution Service.\n'+'Runs: Check_P_Signals\n'+'#'*100)
    #DataReader-------------------------------
    date_from = datetime.strptime(date_from, '%Y-%m-%d')
    date_to = datetime.strptime(date_to, '%Y-%m-%d')
    conf = get_config(config_filename)

    #-----------------------------------------
    #site_dict = load_site_data(conf)
    #P_signal, temp, hum, pre, pua, sea, pm25, pm10 = get_all_data(site_dict, conf, date_from, date_to)
    
    # The anme of col is not important: anyone in ['PM2.5', 'PM10', 'TEMP', 'HUM', 'PRE', 'PUA', 'SAE'] 
    P_signal = load_data_pickle(conf, 'Temp', 'dtonic01')
    P_signal = set_time_period(P_signal, date_from, date_to)
    check_P_signals(P_signal, "PM2.5", conf, date_from, date_to, drawOp=False)

    print('Dtonic Solution Service.\n'+'Runs: Humidity_Analysis\n'+'#'*100)
    #-----------------------------------------
    hum_df = load_data_pickle(conf, 'Humid', 'dtonic01')
    hum_med = load_data_pickle(conf, 'Humid_median', 'dtonic01')
    hum_df = set_time_period(hum_df, date_from, date_to)
    hum_med = set_time_period(hum_med, date_from, date_to)
    hum = (hum_df, hum_med)
    humidity_analysis(hum, conf, date_from, date_to, drawOp=False)

    print('Dtonic Solution Service.\n'+'Runs: Temp_Analysis\n'+'#'*100)
    #-----------------------------------------
    temp_df = load_data_pickle(conf, 'Temp', 'dtonic01')
    temp_med = load_data_pickle(conf, 'Temp_median', 'dtonic01')
    temp_df = set_time_period(temp_df, date_from, date_to)
    temp_med = set_time_period(temp_med, date_from, date_to)
    temp = (temp_df, temp_med)
    temperature_analysis(temp, conf, date_from, date_to, drawOp=False)
    
    print('Dtonic Solution Service.\n'+'Runs: Check Devices\n'+'#'*100)
    #-----------------------------------------
    RunCheckDevices(conf, date_from, date_to, output_filename)

##########################################################################        
#################################AI Model#################################
##########################################################################

#-----------------------------------------------
#	LSTM Model Training/Test
#-----------------------------------------------
def make_dataset(data_dict):

    data_x = []
    data_y = []
    flag =0
    for key in data_dict.keys():
        sample = data_dict[key]
        if np.isnan(sample[0]).any(): continue
        if flag == 0:
            flag = 1
            data_x = sample[0]
            data_y = sample[1]
        else :
            data_x = np.concatenate([data_x,sample[0]],axis=0)
            data_y = np.concatenate([data_y,sample[1]],axis=0)

    print(np.shape(data_x))
    print(np.shape(data_y))

    return data_x, data_y

def train_model(data_x, data_y, conf):

    data_x = data_x/2.0

    y_0 = np.where(data_y == 0)[0]
    y_1 = np.where(data_y == 1)[0]
    y_2 = np.where(data_y == 2)[0]
    min_data_size = min(len(y_0), len(y_1), len(y_2))
    
    x_normal = data_x[y_0]
    x_warn = data_x[y_1]
    x_alarm = data_x[y_2]
    shuffle(x_normal)
    shuffle(x_warn)
    shuffle(x_alarm)
    x_normal = x_normal[:min_data_size]
    x_warn = x_warn[:min_data_size]
    x_alarm = x_alarm[:min_data_size]
    
    x_data = np.concatenate([x_normal,x_warn,x_alarm],axis=0)
    y_data = [0]*len(x_normal)+[1]*len(x_warn)+[2]*len(x_alarm)
    y_data = np.array(y_data)
    y_data = to_categorical(y_data,3)
    
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    
    dev = '/cpu:0'
    opt = Adam(learning_rate=0.01)

    with tf.device(dev):  
        model = Sequential()
        model.add(LSTM(128, input_shape =(8,1)))
        model.add(Dropout(0.3))
        model.add(Dense(3, activation='softmax'))
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

    print(model.summary())
    
    with tf.device(dev):  
        history = model.fit(x_train, y_train, conf.lstm_batch_size, conf.lstm_epochs, validation_data=(x_test,y_test), shuffle=True)
        
    plot_history(history.history)
    plt.show()
    
    model.save(conf.sensor_model_path)
    

    return model, (x_train, y_train), (x_test, y_test)

def test_model(model, X_test, y_test):

    classY = []
    for idx in range(len(y_test)):
        y = y_test[idx]
        y_index = np.argmax(y)
        classY = np.append(classY, y_index)
    realY = classY.astype(np.int32)
    
    pred_Y = model.predict(X_test, verbose=1)
    
    pred_y = []
    for idx in range(len(pred_Y)):
        y = pred_Y[idx]
        y_index = np.argmax(y)
        pred_y = np.append(pred_y, y_index)
    predY = pred_y.astype(np.int32)
    
    cf = confusion_matrix(realY, predY)
    print(cf)
    
    f1w = f1_score(realY, predY, average='weighted', zero_division=1)
    print("f1-score is {}".format(f1w))

 
def load_all_labelled_data(conf):
    with open(conf.pre_label_pickle, 'rb') as f_1:
        data_label_pre = pickle.load(f_1)

    with open(conf.pua_label_pickle, 'rb') as f_1:
        data_label_pua = pickle.load(f_1)

    with open(conf.sea_label_pickle, 'rb') as f_1:
        data_label_sea = pickle.load(f_1)
        
    return data_label_pre, data_label_pua, data_label_sea


#Function for running all codes above
final_temp_list = []

def RunEnkis(args):
    pd.set_option('mode.chained_assignment',  None) # 경고 off
    conf = get_config(args["config_filename"])
    if args["enable_preproc"] == "True":
        if args['start_again']=='True':
            RunPreproc(conf)
        if args["enable_preproc"] == "True":
            if 'location_dict.pickle' not in os.listdir('data/pickles/'): 
                print("Do the first phase of Preprocessing")
                RunPreproc(conf)
            else: 
                print("Do the second phase of Preprocessing")
                RunPreprocWithPrevDict(conf)
        
        print("Splitting Data for Analysis")
        site_dict = load_site_data(conf)
        P_signal, temp, hum, pre, pua, sea = get_all_data(site_dict, conf, args['date_from'],args['date_to'])
        
        print("Making Pickles for Training (SEA, PUA, PRE)")
        GenerateDataSet(args["config_filename"])
        
    else:
        print("Skip Preprocessing")
        #site_dict = load_site_data(conf)
        #P_signal, temp, hum, pre, pua, sea = get_all_data(site_dict, conf, args["date_from"], args["date_to"])
        
    if (args["mode"] == "evaluate"):
        print("Start Analysis")
        RunAllModules(date_from=args['date_from'], date_to=args["date_to"], output_filename=args["output_filename"], config_filename=args["config_filename"])
    else:
        print("Unknown Mode: " + args["mode"])
        
def RunEnkisModelTraining(args):
    pd.set_option('mode.chained_assignment',  None) # 경고 off
    conf = get_config(args["config_filename"])
    conf.lstm_batch_size = 64
    conf.lstm_epochs = 30
    
    if args["enable_preproc"] == "True":
        if args['start_again']=='True':
            RunPreproc(conf)
        if 'location_dict.pickle' not in os.listdir('data/pickles/'): 
            print("Do the first phase of Preprocessing")
            RunPreproc(conf)
        else: 
            print("Do the second phase of Preprocessing")
            RunPreprocWithPrevDict(conf)
            
        print("Making Training Data")
        #-----------------------------------------
        site_dict = load_site_data(conf)
        pre = get_pump_pressure_data(site_dict, conf, args["date_from"], args["date_to"])
        pua = get_pump_current_data(site_dict, conf, args["date_from"], args["date_to"])
        sea = get_sensor_current_data(site_dict, conf, args["date_from"], args["date_to"])
        #-----------------------------------------
    else:
        print("Skip Preprocessing")
        site_dict = load_site_data(conf)
        pre = get_pump_pressure_data(site_dict, conf, args["date_from"], args["date_to"])
        pua = get_pump_current_data(site_dict, conf, args["date_from"], args["date_to"])
        sea = get_sensor_current_data(site_dict, conf, args["date_from"], args["date_to"])

    if (args["mode"] == "train"):
        print("Start Training")
        #-----------------------------------------------
        #	Generate Data Set for training/testing model
        #-----------------------------------------------
        data_label_pre, data_label_pua, data_label_sea = generate_data_set(conf)
        
        #-----------------------------------------------
        #	LSTM Model Training/Test||
        #-----------------------------------------------
        # data_label_pre, data_label_pua, data_label_sea = load_all_labelled_data(conf)
        
        # {{ ENKIS modified
        data_pre_x, data_pre_y = make_dataset(data_label_pre)
        data_pua_x, data_pua_y = make_dataset(data_label_pua)
        data_sea_x, data_sea_y = make_dataset(data_label_sea)
        
        data_all_x = np.concatenate([data_pre_x, data_pua_x, data_sea_x], axis=0)
        data_all_y = np.concatenate([data_pre_y, data_pua_y, data_sea_y], axis=0)

        model, (X_Train, y_train), (X_test, y_test) = train_model(data_all_x, data_all_y, conf)
        # }} end-of-ENKIS-modification

        test_model(model, X_test, y_test)
        
        print("#"*20)
        print("#"*20)
        print("Model_Full_Path: {}".format(conf.sensor_model_path))
        print("#"*20)
        print("#"*20)
        
    else:
        print("Unknown Mode: " + args["mode"])