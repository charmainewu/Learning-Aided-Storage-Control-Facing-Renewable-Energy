# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 19:09:38 2021

@author: jmwu
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import time
from datetime import datetime, date
##################################read data####################################

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
net_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Netload'])

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

for i in range(0,len(data)):
    net_data['Date'][i] = data['Date'][i]
    net_data['Netload'][i] = max(data['Load'][i] - data['Wind'][i],0)
net_data.index = net_data.Date
net_data.drop('Date', axis=1, inplace=True)

renewable_data = renewable_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
price_data = price_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
load_data = load_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
hour_data = hour_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
month_data = month_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
week_data = week_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]
net_data = net_data["2017-01-01 00:00:00":"2018-12-31 23:00:00"]

np.save('./data/renewable.npy', renewable_data) # 注意带上后缀名
np.save('./data/price.npy', price_data) # 注意带上后缀名
np.save('./data/load.npy', load_data) # 注意带上后缀名
np.save('./data/hour.npy', hour_data) # 注意带上后缀名
np.save('./data/month.npy', month_data) # 注意带上后缀名
np.save('./data/week.npy', week_data) # 注意带上后缀名
np.save('./data/net.npy', net_data) # 注意带上后缀名

