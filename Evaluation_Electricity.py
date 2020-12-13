import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import time
from datetime import datetime


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def compare_performance(filePath, outputPath, scaler):
    #将归一化后的数据进行还原
    prediction = pd.read_csv(filePath)
    prediction_inverse = scaler.inverse_transform(prediction)

    for i in range(configure['TargetAttributes']):
        rmse = np.sqrt(mean_squared_error(prediction_inverse[:, i], original_data[:, i]))
        print(str(i) + ' RMSE:' + str(rmse))

    print()

    for i in range(configure['TargetAttributes']):
        mae = mean_absolute_error(original_data[:, i], prediction_inverse[:, i])
        print(str(i) + ' MAE:' + str(mae))

    # 数据概览 - 1
    values = prediction_inverse
    original = original_data

    # 指定要绘制的列
    groups = list(range(configure['TargetAttributes']))
    i = 1
    # 绘制每一列
    plt.figure(figsize=(15, 15))
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(original[:, group])
        plt.plot(values[:, group])
        i += 1
    plt.savefig(outputPath + 'performance_标准.png')
    plt.show()


def compare_performance_noscale(filePath, outputPath):
    prediction = pd.read_csv(filePath).values

    for i in range(configure['TargetAttributes']):
        rmse = np.sqrt(mean_squared_error(prediction[:, i], original_data[:, i]))
        mae = mean_absolute_error(original_data[:, i], prediction[:, i])
        print(str(i) + ' RMSE: ' + str(rmse))
        print(str(i) + ' MAE:  ' + str(mae))
        print()

configure = {}
configure['SourceFilePath'] = './1-Electricity/1-temp.csv'
#configure['InputFilePath'] = './1-Electricity/2-supervisedDataSet.csv'

configure['LSTMPath'] = './1-Electricity/3-LSTM.csv'
configure['FCMPath'] = './1-Electricity/4-FCM.csv'
configure['ANNPath'] = './1-Electricity/5-ANN.csv'
configure['DFCMPath'] = './1-Electricity/6-DFCM.csv'
configure['ARIMAPath'] = './1-Electricity/7-ARIMA.csv'
configure['WaveletHFCMPath'] = './1-Electricity/8-Wavelet-HFCM.csv'
configure['VARPath'] = './1-Electricity/9-VAR.csv'
configure['DFCM_3Path'] = './1-Electricity/10-DFCM_3.csv'
configure['DFCM_F3Path'] = './1-Electricity/11-DFCM_F3.csv'
configure['DFCM_F1Path'] = './1-Electricity/12-DFCM_F1.csv'
configure['LSTM_sPath'] = './1-Electricity/13-LSTM_s.csv'
configure['ARIMA_NoAutoPath'] = './1-Electricity/14-ARIMA_NoAuto.csv'

configure['LSTMPltPath'] = './1-Electricity/3-LSTM/'
configure['FCMPltPath'] = './1-Electricity/4-FCM/'
configure['ANNPltPath'] = './1-Electricity/5-ANN/'
configure['DFCMPltPath'] = './1-Electricity/6-DFCM/'
configure['ARIMAPltPath'] = './1-Electricity/7-ARIMA/'
configure['WaveletHFCMPltPath'] = './1-Electricity/8-Wavelet-HFCM/'
configure['VARPltPath'] = './1-Electricity/9-VAR/'
configure['DFCM_3PltPath'] = './1-Electricity/10-DFCM_3.csv'
configure['DFCM_F3PltPath'] = './1-Electricity/11-DFCM_F3/'
configure['DFCM_F1PltPath'] = './1-Electricity/12-DFCM_F1/'
configure['LSTM_sPltPath'] = './1-Electricity/13-LSTM_s/ '
configure['LSTM_sPath'] = './1-Electricity/13-LSTM_s.csv'
configure['ARIMA_NoAutoPltPath'] = './1-Electricity/14-ARIMA_NoAuto/'

configure['AllAttributes'] = 8
configure['TargetAttributes'] = 3
configure['InputAttributes'] = [0,1,2,3,4,5,6,7]
configure['OutputAttributes'] = [5,6,7]
configure['Length'] = 21899


n_train = int(0.7 * configure['Length'])

original_data  = pd.read_csv(configure['SourceFilePath'])
del original_data['Date_Time']

original_data_all = original_data.values[:,configure['OutputAttributes']]
original_data = original_data_all[n_train + 1:,:]

scaler = preprocessing.StandardScaler()
scaled = scaler.fit_transform(original_data_all)

compare_performance(configure['DFCMPath'],configure['DFCMPltPath'],scaler)