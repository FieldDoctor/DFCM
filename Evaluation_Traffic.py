import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

import time
from datetime import datetime


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def compare_performance(filePath, outputPath, scaler):
    prediction = pd.read_csv(filePath)
    prediction_inverse = scaler.inverse_transform(prediction)

    for i in range(configure['TargetAttributes']):
        rmse = np.sqrt(mean_squared_error(prediction_inverse[:, i], original_data[:, i]))
        mape = mean_absolute_percentage_error(original_data[:, i], prediction_inverse[:, i])
        print(i, rmse)
        print(i, mape)

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
        mape = mean_absolute_percentage_error(original_data[:, i], prediction[:, i])
        print(i, rmse)
        print(i, mape)

configure = {}
configure['SourceFilePath'] = './5-Traffic/1-temp.csv'

configure['LSTMPath'] = './5-Traffic/3-LSTM.csv'
configure['FCMPath'] = './5-Traffic/4-FCM.csv'
configure['ANNPath'] = './5-Traffic/5-ANN.csv'
configure['DFCMPath'] = './5-Traffic/6-DFCM.csv'
configure['ARIMAPath'] = './5-Traffic/7-ARIMA.csv'
configure['WaveletHFCMPath'] = './5-Traffic/8-Wavelet-HFCM.csv'
configure['VARPath'] = './5-Traffic/9-VAR.csv'

configure['LSTMPltPath'] = './5-Traffic/3-LSTM/'
configure['FCMPltPath'] = './5-Traffic/4-FCM/'
configure['ANNPltPath'] = './5-Traffic/5-ANN/'
configure['DFCMPltPath'] = './5-Traffic/6-DFCM/'
configure['ARIMAPltPath'] = './5-Traffic/7-ARIMA/'
configure['WaveletHFCMPltPath'] = './5-Traffic/8-Wavelet-HFCM/'
configure['VARPltPath'] = './5-Traffic/9-VAR/'

configure['AllAttributes'] = 7
configure['TargetAttributes'] = 6
configure['InputAttributes'] = [0,1,2,3,4,5,6]
configure['OutputAttributes'] = [1,2,3,4,5,6]
configure['Length'] = 3602

n_train = int(0.7 * configure['Length'])

original_data  = pd.read_csv(configure['SourceFilePath'])

original_data = original_data.values
original_data = original_data[n_train + 1:,configure['OutputAttributes']]

# z-score
scaler = preprocessing.StandardScaler()
scaled = scaler.fit_transform(original_data)

compare_performance(configure['DFCMPath'],configure['DFCMPltPath'],scaler)