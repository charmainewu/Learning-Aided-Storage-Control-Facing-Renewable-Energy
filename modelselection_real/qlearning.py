# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:29:44 2021

@author: jmwu
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.utils.random import sample_without_replacement
from benchmark_q import BenchmarkInAccurateDemand
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('simulation.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d %H:%M')
df.index = df['Date']
data = df.sort_index(ascending=True, axis=0)

price_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Price'])
renewable_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Wind'])
load_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Load'])
hour_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Hour'])
week_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Week'])
month_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Month'])

for i in range(0,len(data)):
    price_data['Date'][i] = data['Date'][i]
    price_data['Price'][i] = max(0,data['Price'][i])
price_data.index = price_data.Date
price_data.drop('Date', axis=1, inplace=True)
for i in range(0,len(data)):
    renewable_data['Date'][i] = data['Date'][i]
    renewable_data['Wind'][i] = data['Wind'][i]
renewable_data.index = renewable_data.Date
renewable_data.drop('Date', axis=1, inplace=True)
for i in range(0,len(data)):
    load_data['Date'][i] = data['Date'][i]
    load_data['Load'][i] = data['Load'][i]
load_data.index = load_data.Date
load_data.drop('Date', axis=1, inplace=True)
for i in range(0,len(data)):
    hour_data['Date'][i] = data['Date'][i]
    if data['Date'][i].hour<6:
        hour_data['Hour'][i] = 1
    elif data['Date'][i].hour>=6 and data['Date'][i].hour<12:
        hour_data['Hour'][i] = 2
    elif data['Date'][i].hour>=12 and data['Date'][i].hour<18:
        hour_data['Hour'][i] = 3
    else:
        hour_data['Hour'][i] = 4
hour_data.index = hour_data.Date
hour_data.drop('Date', axis=1, inplace=True)
for i in range(0,len(data)):
    month_data['Date'][i] = data['Date'][i]
    month_data['Month'][i] = data['Date'][i].month
month_data.index = month_data.Date
month_data.drop('Date', axis=1, inplace=True)

for i in range(0,len(data)):
    week_data['Date'][i] = data['Date'][i]
    week_data['Week'][i] = data['Date'][i].weekday()
week_data.index = week_data.Date
week_data.drop('Date', axis=1, inplace=True)

renewable_data = renewable_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
price_data = price_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
load_data = load_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
hour_data = hour_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
month_data = month_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
week_data = week_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]

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

###########################################################################

Demand = load_data.values.reshape(len(load_data))
Price = price_data.values.reshape(len(price_data))
Renewables = renewable_data.values.reshape(len(renewable_data))

Window = 50;
Capacity = 100000; 
Ntrain = int(len(Demand) * 0.50); 
Ntest = int(len(Demand) * 0.20); 
Nvalid = int(len(Demand) * 0.30); 

n_iter = 20
epochs = [100,500,1000,5000]
learning_rate = [0.0001,0.001,0.01]
reward_decay = [0.7,0.8,0.9]
e_greedy = [0.7,0.8,0.9]
vanish = [0.7,0.8,0.9]

param_grid = dict(epochs=epochs, learning_rate=learning_rate, reward_decay=reward_decay, e_greedy = e_greedy,vanish=vanish)
param_com = list(ParameterGrid(param_grid))

result = {}; count = 0;
for i in sample_without_replacement(len(param_com), n_iter, random_state=5):
    
    epochs_sample = param_com[i]['epochs']
    learning_rate_sample = param_com[i]['learning_rate']
    e_greedy_sample = param_com[i]['e_greedy']
    reward_decay_sample = param_com[i]['reward_decay']
    vanish_sample = param_com[i]['vanish']
    
    BAD = BenchmarkInAccurateDemand(Capacity,Window, Ntrain,Ntest,Nvalid, Price,Demand, Renewables,learning_rate_sample,reward_decay_sample,e_greedy_sample,vanish_sample,epochs_sample)
    cost_rl= BAD.rl()
    cost_ofl = BAD.ofl()
    acc = cost_rl/cost_ofl
    
    result[count] = {}
    result[count]["acc"],result[count]["par"] = acc,[learning_rate_sample,reward_decay_sample,e_greedy_sample,vanish_sample,epochs_sample]
    
    count = count + 1 
np.save('./data/qlearning_grid.npy', result) # 注意带上后缀名
#result = np.load('windowprice.npy',allow_pickle=True).item()
