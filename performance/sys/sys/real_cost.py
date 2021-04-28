# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:45:11 2020

@author: jmwu
"""

from benchmark import BenchmarkAccurateDemand
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn import mixture
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
from keras.models import load_model
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import seaborn as sns

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

scaler_l = MinMaxScaler(feature_range=(0, 1))
l = scaler_l.fit_transform(net_data)
l = scaler_l.inverse_transform(l)/10000

scaler_e = MinMaxScaler(feature_range=(0, 1))
e = scaler_e.fit_transform(price_data)
e = scaler_e.inverse_transform(e)/100

dataset_r  = np.hstack((r,w,h))
dataset_p  = np.hstack((p,w,h))

###########################################################################

Demand = load_data.reshape(len(load_data))
Price = price_data.reshape(len(price_data))
Renewables = renewable_data.reshape(len(renewable_data))
PriceScalar = scaler_p
WindScalar = scaler_r
WindStackData = dataset_r
PriceStackData = dataset_p

Interval = 24;  cr = 1
Window = Interval;
Capacity = int(max(l)*10000*cr);
Ntrain = int(len(dataset_r) * 0.50);
Ntest = int(len(dataset_r) * 0.10/24)*24;
Nvalid = int(len(dataset_r) * 0.30);
Nvalid = int(len(dataset_r) * 0.30);
print("end")

singleprice_grid = np.load('./data/singleprice_best1.npy',allow_pickle=True).astype(int)
singlewind_grid = np.load('./data/singlewind_best1.npy',allow_pickle=True).astype(int)

s2sprice_grid24 = np.load('./data/s2sprice_best24.npy',allow_pickle=True).astype(int)
s2swind_grid24 = np.load('./data/s2swind_best24.npy',allow_pickle=True).astype(int)

vectorprice_grid24 = np.load('./data/windowprice_best24.npy',allow_pickle=True).astype(int)
vectorwind_grid24 = np.load('./data/windowwind_best24.npy',allow_pickle=True).astype(int)

Backday = [singlewind_grid[5],singleprice_grid[5]]
##########################################################
LSTM_TYPE = "Point"
WindModel = load_model("./winddata/singlewind.h5")
PriceModel = load_model("./pricedata/singleprice.h5")

BAD = BenchmarkAccurateDemand(Capacity, Window, Ntrain, Ntest, Nvalid, Backday, Interval, Price, Demand, Renewables, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData,LSTM_TYPE)

cost_mpc= BAD.mpc()
np.save("./data/cost_mpc.npy",cost_mpc)
print("endmpc")
cost_thb= BAD.thb()
np.save("./data/cost_thb.npy",cost_thb)
print("endthb")
cost_ofl = BAD.ofl()
np.save("./data/cost_ofl.npy",cost_ofl)
print("endofl")
cost_nos = BAD.nos()
np.save("./data/cost_nos.npy",cost_nos)
print("endnos")
cost_deta= BAD.deta()
np.save("./data/cost_deta.npy",cost_deta)
print("enddeta")

cost_peta = BAD.leta()
np.save("./data/cost_peta.npy",cost_peta)
print("endpeta")

##########################################################
Backday = [s2swind_grid24[5],s2sprice_grid24[5]]

LSTM_TYPE = "S2S"
WindModel = Seq2Seq(output_dim=1, hidden_dim=int(s2swind_grid24[0]), output_length=24, input_shape=(Backday[0], dataset_r.shape[1]), peek=False, depth=int(s2swind_grid24[1]), dropout = 0.2)

PriceModel = Seq2Seq(output_dim=1, hidden_dim=int(s2sprice_grid24[0]), output_length=24, input_shape=(Backday[1], dataset_r.shape[1]), peek=False, depth=int(s2sprice_grid24[1]), dropout = 0.2)

WindModel.load_weights("./winddata/s2swind24.h5")
PriceModel.load_weights("./pricedata/s2sprice24.h5")

BAD = BenchmarkAccurateDemand(Capacity, Window, Ntrain, Ntest, Nvalid, Backday, Interval, Price, Demand, Renewables, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData,LSTM_TYPE)
cost_seta = BAD.leta()
np.save("./data/cost_seta.npy",cost_seta)
print("endseta")

##########################################################
Backday = [vectorwind_grid24[5],vectorprice_grid24[5]]

LSTM_TYPE = "Vector"

WindModel = load_model("./winddata/windowwind24.h5")
PriceModel = load_model("./pricedata/windowprice24.h5")

BAD = BenchmarkAccurateDemand(Capacity, Window, Ntrain, Ntest, Nvalid, Backday, Interval, Price, Demand, Renewables, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData,LSTM_TYPE)
cost_veta = BAD.leta()
np.save("./data/cost_veta.npy",cost_veta)
print("endveta")










