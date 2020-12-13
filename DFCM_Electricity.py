configure = {}
configure['SourceFilePath'] = './1-Electricity/1-temp.csv'
configure['InputFilePath'] = './1-Electricity/2plus-supervisedDataSet_zscore.csv'
configure['OutputFilePath'] = './1-Electricity/6-DFCM.csv'
configure['PltFilePath'] = './1-Electricity/6-DFCM/'
configure['AllAttributes'] = 8
configure['TargetAttributes'] = 3
configure['InputAttributes'] = [1,2,3,4,5,6,7]
configure['OutputAttributes'] = [13,14,15]
configure['TimeAttributes'] = [0]
configure['Length'] = 21899

configure['global_epochs'] = 400

configure['f_batch_size'] = 25000
configure['f_epochs'] = 15
configure['hidden_layer'] = 10

configure['n_batch_size'] = 25000
configure['n_epochs'] = 15
configure['LSTM_hiddenDim'] = 15


import os
import time

from pandas import DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras import optimizers

#optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

global_epochs = configure['global_epochs']

f_batch_size = configure['f_batch_size']
f_epochs = configure['f_epochs']
hidden_layer = configure['hidden_layer']

n_batch_size = configure['n_batch_size']
n_epochs = configure['n_epochs']
LSTM_hiddenDim = configure['LSTM_hiddenDim']

# 函数 - sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False

mkdir(configure['PltFilePath'])


# 加载数据集
dataset = pd.read_csv(configure['InputFilePath'])

# 狗造训练集（70%）和测试集（30%）
values = dataset.values
n_train = int(0.7 * configure['Length'])

train = values[:n_train, :]
test = values[n_train:, :]

train_X, train_Y = sigmoid( train[:, configure['InputAttributes']] ) , train[:,configure['OutputAttributes']]
test_X, test_Y = sigmoid( test[:, configure['InputAttributes']] ), test[:,configure['OutputAttributes']]

train_U = train[:,configure['TimeAttributes']]
train_U = train_U.reshape((train_U.shape[0], 1, train_U.shape[1]))
test_U = test[:,configure['TimeAttributes']]
test_U = test_U.reshape((test_U.shape[0], 1, test_U.shape[1]))

print('Train dataset length : ' + str(len(train)) + '.')
print('Test dataset length : ' + str(len(test)) + '.')
print('------')
print('X dim : ' + str(train_X.shape[1]) + '.')
print('Y dim : ' + str(train_Y.shape[1]) + '.')
print('------')
print('train_X shape : ' + str(train_X.shape))
print('train_Y shape : ' + str(train_Y.shape))
print('train_U shape : ' + str(train_U.shape))
print('------')
print('test_X shape : ' + str(test_X.shape))
print('test_Y shape : ' + str(test_Y.shape))
print('test_U shape : ' + str(test_U.shape))

# 设计DFCM_3网络
model_f = [0 for i in range(len(configure['OutputAttributes']))]
for i in range(len(configure['OutputAttributes'])):
    model_f[i] = Sequential()
    model_f[i].add(Dense(hidden_layer, input_dim=train_X.shape[1], activation='relu', use_bias=False))
    #model_f[i].add(Dense(hidden_layer, input_dim=hidden_layer, activation='relu', use_bias=False))
    #model_f[i].add(Dense(hidden_layer, input_dim=hidden_layer, activation='relu', use_bias=False))
    model_f[i].add(Dense(1, input_dim=hidden_layer, use_bias=False))
    model_f[i].compile(loss='mean_squared_error', optimizer='adam')

model_u = [0 for i in range(len(configure['OutputAttributes']))]
for i in range(len(configure['OutputAttributes'])):
    model_u[i] = Sequential()
    model_u[i].add(LSTM(LSTM_hiddenDim, input_shape=(train_U.shape[1], train_U.shape[2])))
    model_u[i].add(Dense(1, input_dim=LSTM_hiddenDim, use_bias=True))
    model_u[i].compile(loss='mean_squared_error', optimizer='adam')

for i in range(global_epochs):
    start = time.time()
    if i == 0:
        y_f = train_Y
    else:
        y_f = train_Y - y_u_predict

    for j in range(len(configure['OutputAttributes'])):
        model_f[j].fit(train_X, y_f[:, j], f_batch_size, f_epochs, verbose=0, shuffle=False)

    y_f_predict = DataFrame()
    for j in range(len(configure['OutputAttributes'])):
        y_f_predict[str(j)] = model_f[j].predict(train_X).reshape(-1)
    y_f_predict = y_f_predict.values

    y_u = train_Y - y_f_predict
    # for j in range(len(configure['OutputAttributes'])):
    # print('f' + str(j + 1) + ' : ' + str(model_f[j].evaluate(train_X,y_f[:,j],verbose=2)))
    # print('The ' + str(i + 1) + ' times f() training finished.  loss:' + str( pow(abs(y_u), 2).mean().mean() ))

    for j in range(len(configure['OutputAttributes'])):
        model_u[j].fit(train_U, y_u[:, j], n_batch_size, n_epochs, verbose=0)

    y_u_predict = DataFrame()
    for j in range(len(configure['OutputAttributes'])):
        y_u_predict[str(j)] = model_u[j].predict(train_U).reshape(-1)
    y_u_predict = y_u_predict.values

    # for j in range(len(configure['OutputAttributes'])):
    # print('u' + str(j + 1) + ' : ' + str(model_u[j].evaluate(train_U, y_u[:,j],verbose=2)))
    # print('The ' + str(i + 1) + ' times u() training finished.  loss:' + str( pow(abs(train_Y - y_u_predict), 2).mean().mean() ))

    # evaluate
    yhat_f_predict = DataFrame()
    for j in range(len(configure['OutputAttributes'])):
        yhat_f_predict[str(j)] = model_f[j].predict(test_X).reshape(-1)
    yhat_f_predict = yhat_f_predict.values

    yhat_u_predict = DataFrame()
    for j in range(len(configure['OutputAttributes'])):
        yhat_u_predict[str(j)] = model_u[j].predict(test_U).reshape(-1)
    yhat_u_predict = yhat_u_predict.values

    predict_train = y_u_predict + y_f_predict
    predict_test = yhat_u_predict + yhat_f_predict
    real_train = train_Y
    real_test = test_Y

    error_train = pow(abs(real_train - predict_train), 2)
    error_test = pow(abs(real_test - predict_test), 2)
    # print('The ' + str(i + 1) + ' times train error:    ' + str(error_train.mean().mean()))
    # print('The ' + str(i + 1) + ' times test error:    ' + str(error_test.mean().mean()))
    print(i + 1, error_train.mean().mean(), error_test.mean().mean())

    if (error_test.mean().mean() < 0.125):
        break

    # print('This epoch TimeCost:' + str(time.time()-start) + 's.')

# 预测 & 输出
yhat_f_predict = DataFrame()
for j in range(len(configure['OutputAttributes'])):
    yhat_f_predict[str(j)] = model_f[j].predict(test_X).reshape(-1)
yhat_f_predict = yhat_f_predict.values

yhat_u_predict = DataFrame()
for j in range(len(configure['OutputAttributes'])):
    yhat_u_predict[str(j)] = model_u[j].predict(test_U).reshape(-1)
yhat_u_predict = yhat_u_predict.values

yhat = yhat_u_predict + yhat_f_predict
DataFrame(yhat).to_csv(configure['OutputFilePath'],index=False)

for j in range(len(configure['OutputAttributes'])):
    model_f[j].save(configure['PltFilePath'] + 'model_f_' + str(j+1) + '.h5')
    model_u[j].save(configure['PltFilePath'] + 'model_u_' + str(j+1) + '.h5')

# 数据概览 - 1
values = yhat
original = test_Y

# 指定要绘制的列
groups = list(range(configure['TargetAttributes']))
i = 1
# 绘制每一列
plt.figure(figsize=(15,15))
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(original[:, group])
    plt.plot(values[:, group])
    i += 1
plt.savefig(configure['PltFilePath'] + 'performance.png')
plt.show()