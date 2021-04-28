# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 00:32:34 2021

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

dataset_p  = np.hstack((p,w,h))
train_size = int(len(dataset_p) * 0.50)
valid_size = int(len(dataset_p) * 0.30)
test_size = int(len(dataset_p) * 0.20)
train, valid, test = dataset_p[0:train_size,:], dataset_p[train_size:train_size+valid_size,:], dataset_p[train_size+valid_size:train_size+valid_size+test_size,:]


def lstmmodel(neurons,layer,drop,batch_size,epochs,backday,n_outputs):
  x_train, y_train = [], []
  x_valid, y_valid = [], []
  x_test, y_test = [], []
  
  for i in range(backday,train_size-n_outputs):
      x_train.append(train[i-backday:i,:])
      y_train.append(train[i:i+n_outputs,0])
  x_train, y_train = np.array(x_train), np.array(y_train)
  y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
  
  for i in range(backday,valid_size-n_outputs):
      x_valid.append(valid[i-backday:i,:])
      y_valid.append(valid[i:i+n_outputs,0])
  x_valid, y_valid = np.array(x_valid), np.array(y_valid)
  y_valid = y_valid.reshape((y_valid.shape[0], y_valid.shape[1], 1))
  
  for i in range(backday,test_size-n_outputs):
      x_test.append(test[i-backday:i,:])
      y_test.append(test[i:i+n_outputs,0])
  x_test, y_test = np.array(x_test), np.array(y_test)
  y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
  
  print(neurons,layer,drop,batch_size,epochs,backday,n_outputs)
  model = Seq2Seq(output_dim=1, hidden_dim=neurons, output_length=n_outputs, input_shape=(x_train.shape[1], x_train.shape[2]), peek=False, depth=layer,dropout=drop)
  model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
  early_stopping = EarlyStopping(monitor='mean_absolute_error', patience=10)
  model.fit(x_train, y_train, epochs=epochs, batch_size = len(x_train), validation_data=(x_valid, y_valid), verbose=0, callbacks=[early_stopping])
  
  loss,accuracy = model.evaluate(x_test,y_test)
  print('Test Mean Absolute Error: %f using %f,%f,%f,%f,%f,%f' % (accuracy,neurons,layer,drop,batch_size,epochs,backday))

  return model,accuracy,[neurons,layer,drop,batch_size,epochs,backday]

n_iter = 10
backday = [6,12,24]
neurons = [16,64,128]
layer = [1,2,3]
batch_size = [train_size]
drop = [0.2]
epochs = [100,500,1000,5000]

param_grid = dict(batch_size=batch_size, neurons=neurons, layer=layer, drop = drop, epochs=epochs,backday=backday)
param_com = list(ParameterGrid(param_grid))

for n_outputs in [6,12,24]:
    result = {}; count = 0; acc_best=1000;
    for i in sample_without_replacement(len(param_com), n_iter, random_state=5):
      batch_size_sample = param_com[i]['batch_size']
      neurons_sample = param_com[i]['neurons']
      layer_sample = param_com[i]['layer']
      drop_sample = param_com[i]['drop']
      epochs_sample  = param_com[i]['epochs']
      backday_sample  = param_com[i]['backday']
      result[count] = {}
      model,result[count]["acc"],result[count]["par"] = lstmmodel(neurons_sample,layer_sample,drop_sample,batch_size_sample,epochs_sample,backday_sample,n_outputs)
      
      if result[count]["acc"] < acc_best:
        best = result[count]["par"]
        acc_best = result[count]["acc"]
        model.save('./data/s2sprice'+str(n_outputs)+'.h5')
      
      count = count + 1
    np.save('./data/s2sprice_grid'+str(n_outputs)+'.npy', result) # 注意带上后缀名
    #result = np.load('windowprice.npy',allow_pickle=True).item()

