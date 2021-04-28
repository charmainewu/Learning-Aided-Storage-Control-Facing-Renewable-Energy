# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:06:15 2021

@author: jmwu
"""

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import time
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import ParameterGrid
from sklearn.utils.random import sample_without_replacement
##################################read data####################################

renewable_data = np.load('./data/renewable.npy',allow_pickle=True) 
price_data = np.load('./data/price.npy',allow_pickle=True) 
load_data= np.load('./data/load.npy',allow_pickle=True) 
hour_data = np.load('./data/hour.npy',allow_pickle=True) 
month_data = np.load('./data/month.npy',allow_pickle=True) 
week_data = np.load('./data/week.npy',allow_pickle=True) 
net_data = np.load('./data/net.npy',allow_pickle=True) 

scaler_r = MinMaxScaler(feature_range=(0, 1))
r = scaler_r.fit_transform(renewable_data)

scaler_p = MinMaxScaler(feature_range=(0, 1))
p = scaler_p.fit_transform(price_data)

enc_h = OneHotEncoder(handle_unknown='ignore')
h = enc_h.fit_transform(hour_data).toarray()

enc_m = OneHotEncoder(handle_unknown='ignore')
m = enc_m.fit_transform(month_data).toarray()

enc_w = OneHotEncoder(handle_unknown='ignore')
w = enc_w.fit_transform(week_data).toarray()

##################################renewable pre################################

dataset_p  = np.hstack((r,w,h))
train_size = int(len(dataset_p) * 0.50)
valid_size = int(len(dataset_p) * 0.30)
test_size = int(len(dataset_p) * 0.20)
train, valid, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+valid_size,:], dataset_p[train_size+valid_size:train_size+valid_size+test_size,:]

###############################Back############################################

def lstmmodel(neurons,layer,drop,batch_size,epochs,backday):
  x_train, y_train = [], []
  x_valid, y_valid = [], []
  x_test, y_test = [], []
  
  for i in range(backday,train_size):
      x_train.append(train[i-backday:i,:])
      y_train.append(train[i,0])
  x_train, y_train = np.array(x_train), np.array(y_train)
  
  for i in range(backday,valid_size):
      x_valid.append(valid[i-backday:i,:])
      y_valid.append(valid[i,0])
  x_valid, y_valid = np.array(x_valid), np.array(y_valid)
  
  for i in range(backday,test_size):
      x_test.append(test[i-backday:i,:])
      y_test.append(test[i,0])
  x_test, y_test = np.array(x_test), np.array(y_test)
  
  model = Sequential()
  if layer == 1:
      model.add(LSTM(units=neurons, return_sequences=False, input_shape=(x_train.shape[1],x_train.shape[2])))
      model.add(Dropout(drop))
      model.add(Dense(1))
      model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
  if layer >= 2:
      model.add(LSTM(units=neurons, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
      model.add(Dropout(drop))
      for i in range(layer-2):
          model.add(LSTM(units=neurons, return_sequences=True))
          model.add(Dropout(drop))
      model.add(LSTM(units=neurons, return_sequences=False))
      model.add(Dropout(drop))
      model.add(Dense(1))
      model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
  early_stopping = EarlyStopping(monitor='mean_absolute_error', patience=10)
  model.fit(x_train, y_train, epochs=epochs, batch_size = len(x_train), validation_data=(x_valid, y_valid), verbose=0, callbacks=[early_stopping])

  loss,accuracy = model.evaluate(x_test,y_test)
  print('Test Mean Absolute Error: %f using %f,%f,%f,%f,%f,%f' % (accuracy,neurons,layer,drop,batch_size,epochs,backday))

  return model,accuracy,[neurons,layer,drop,batch_size,epochs,backday]

def fitmodel(neurons,layer,drop,batch_size,epochs,backday):
  x_train, y_train = [], []
  x_valid, y_valid = [], []
  x_test, y_test = [], []
  
  for i in range(backday,train_size):
      x_train.append(train[i-backday:i,:])
      y_train.append(train[i,0])
  x_train, y_train = np.array(x_train), np.array(y_train)
  
  for i in range(backday,valid_size):
      x_valid.append(valid[i-backday:i,:])
      y_valid.append(valid[i,0])
  x_valid, y_valid = np.array(x_valid), np.array(y_valid)
  
  for i in range(backday,test_size):
      x_test.append(test[i-backday:i,:])
      y_test.append(test[i,0])
  x_test, y_test = np.array(x_test), np.array(y_test)
  
  model = Sequential()
  if layer == 1:
      model.add(LSTM(units=neurons, return_sequences=False, input_shape=(x_train.shape[1],x_train.shape[2])))
      model.add(Dropout(drop))
      model.add(Dense(1))
      model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
  if layer >= 2:
      model.add(LSTM(units=neurons, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
      model.add(Dropout(drop))
      for i in range(layer-2):
          model.add(LSTM(units=neurons, return_sequences=True))
          model.add(Dropout(drop))
      model.add(LSTM(units=neurons, return_sequences=False))
      model.add(Dropout(drop))
      model.add(Dense(1))
      model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
  early_stopping = EarlyStopping(monitor='mean_absolute_error', patience=10)
  model.fit(x_train, y_train, epochs=epochs, batch_size = len(x_train), validation_data=(x_valid, y_valid), verbose=0, callbacks=[early_stopping])
  model.save("./data/singlewind.h5")


n_iter = 10
backday = [6,12,24]
neurons = [16, 64, 128]
layer = [1,2,3]
batch_size = [train_size]
drop = [0.2]
epochs = [100,500,1000,5000]
n_outputs = 1

param_grid = dict(batch_size=batch_size, neurons=neurons, layer=layer, drop = drop, epochs=epochs,backday=backday)
param_com = list(ParameterGrid(param_grid))

result = {}; count = 0; acc_best = 10000
for i in sample_without_replacement(len(param_com), n_iter, random_state=5):
    batch_size_sample = param_com[i]['batch_size']
    neurons_sample = param_com[i]['neurons']
    layer_sample = param_com[i]['layer']
    drop_sample = param_com[i]['drop']
    epochs_sample  = param_com[i]['epochs']
    backday_sample  = param_com[i]['backday']
    result[count] = {}
    model, result[count]["acc"],result[count]["par"] = lstmmodel(neurons_sample,layer_sample,drop_sample,batch_size_sample,epochs_sample,backday_sample)
    if result[count]["acc"] < acc_best:
      best = result[count]["par"]
      backday_best= best[-1]
      acc_best = result[count]["acc"]
      model.save("./data/singlewind.h5")
    count = count + 1 
np.save('./data/singlewind_grid.npy', result) # 注意带上后缀名
#result = np.load('windowwind.npy',allow_pickle=True).item()

#fitmodel(best[:])
model = load_model("./data/singlewind.h5")

mae = []; 
for n_outputs in [6,12,24]:
  test_predict = []; test_y = [];
  for i in range(backday_best,test_size-n_outputs):
      test_copy = test.copy()
      test_predict_a = []
      for j in range(n_outputs):
          x_test = test_copy[i+j-backday_best:i+j,:]
          x_test = np.reshape(x_test, (1,x_test.shape[0],x_test.shape[1]))
          test_predict_v = model.predict(x_test)
          test_copy[i+j,0] = test_predict_v[0][0]
          test_predict_a.append(test_predict_v[0][0])
      test_predict.append(test_predict_a)
      test_y.append(test[i:i+n_outputs,0])
  test_predict = np.array(test_predict); test_y = np.array(test_y)
  mae.append(mean_absolute_error(test_y, test_predict))
np.save('./data/singlewind_mae.npy', mae)
