# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 23:28:40 2021

@author: jmwu
"""


from benchmark import BenchmarkInAccurateDemand
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model
import torch
from torch import nn
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import warnings
warnings.filterwarnings("ignore")


class lstm(nn.Module):
    def __init__(self,input_size=12,hidden_size=4,output_size=1,num_layer=1,seq_length=1):
        super(lstm,self).__init__()
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer,dropout=0.2)
        self.layer2 = nn.Linear(hidden_size*seq_length,output_size)
        
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = torch.transpose(x,0,1)
        x = x.contiguous().view(b,h*s)
        x = self.layer2(x)
        x = x.view(b,-1)
        return x

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
DataSet = np.hstack((l,e,w,h))
Demand = load_data.reshape(len(load_data))
Price = price_data.reshape(len(price_data))
Renewables = renewable_data.reshape(len(renewable_data))
PriceScalar = scaler_p
WindScalar = scaler_r
WindStackData = dataset_r
PriceStackData = dataset_p

Interval = 24
cr = 1; Capacity = int(max(l)*10000*cr); 
Ntrain = int(len(dataset_r) * 0.50); 
Ntest = int(len(dataset_r) * 0.10/24)*24;
Nvalid = int(len(dataset_r) * 0.30);

singleprice_grid = np.load('./data/singleprice_best1.npy',allow_pickle=True).astype(int)
singlewind_grid = np.load('./data/singlewind_best1.npy',allow_pickle=True).astype(int) 

s2sprice_grid24 = np.load('./data/s2sprice_best24.npy',allow_pickle=True).astype(int) 
s2swind_grid24 = np.load('./data/s2swind_best24.npy',allow_pickle=True).astype(int)
 
vectorprice_grid24 = np.load('./data/windowprice_best24.npy',allow_pickle=True).astype(int)
vectorwind_grid24 = np.load('./data/windowwind_best24.npy',allow_pickle=True).astype(int)

end_grid24_05 = np.load('./enddata/endlearning_best240.5.npy',allow_pickle=True).astype(int) 
end_grid24_1 = np.load('./enddata/endlearning_best241.npy',allow_pickle=True).astype(int) 
end_grid24_5 = np.load('./enddata/endlearning_best245.npy',allow_pickle=True).astype(int) 

##########################################################
Backday = [end_grid24_1[1],end_grid24_1[1]]
##########################################################
device = torch.device("cpu")
LSTM_TYPE = "End"
EndModel = lstm(13,int(end_grid24_1[2]),24,int(end_grid24_1[3]),int(end_grid24_1[1]))
EndModel.load_state_dict(torch.load("./enddata/endtoend241.pth",device))
EndModel = EndModel.to(device)
WindModel = load_model("./winddata/singlewind.h5")
PriceModel = load_model("./pricedata/singleprice.h5")
BAD = BenchmarkInAccurateDemand(Capacity, Ntrain, Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, EndModel, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData, LSTM_TYPE, DataSet)
cost_eeta = BAD.eeta()
np.save("./data/bt1/cost_eeta.npy",cost_eeta)
print("end")

##########################################################
Backday = [singlewind_grid[5],singleprice_grid[5]]
LSTM_TYPE = "Point"
WindModel = load_model("./winddata/singlewind.h5")
PriceModel = load_model("./pricedata/singleprice.h5")
BAD = BenchmarkInAccurateDemand(Capacity, Ntrain, Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, EndModel, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData, LSTM_TYPE, DataSet)
cost_deta= BAD.deta()
np.save("./data/bt1/cost_deta.npy",cost_deta)
print("end")
cost_mpc= BAD.mpc()
np.save("./data/bt1/cost_mpc.npy",cost_mpc)
print("end")
cost_thb= BAD.thb()
np.save("./data/bt1/cost_thb.npy",cost_thb)
print("end")
cost_ofl = BAD.ofl()
np.save("./data/bt1/cost_ofl.npy",cost_ofl)
print("end")
cost_peta = BAD.leta()
np.save("./data/bt1/cost_peta.npy",cost_peta)
print("end")
cost_nos = BAD.nos()
np.save("./data/bt1/cost_nos.npy",cost_nos)
print("end")

##########################################################
Backday = [s2swind_grid24[5],s2swind_grid24[5]]
LSTM_TYPE = "S2S"
WindModel = Seq2Seq(output_dim=1, hidden_dim=int(s2swind_grid24[0]), output_length=24, input_shape=(Backday[0], dataset_r.shape[1]), peek=False, depth=int(s2swind_grid24[1]), dropout = 0.2)

PriceModel = Seq2Seq(output_dim=1, hidden_dim=int(s2sprice_grid24[0]), output_length=24, input_shape=(Backday[1], dataset_r.shape[1]), peek=False, depth=int(s2sprice_grid24[1]), dropout = 0.2)

WindModel.load_weights("./winddata/s2swind24.h5")
PriceModel.load_weights("./pricedata/s2sprice24.h5")

BAD = BenchmarkInAccurateDemand(Capacity, Ntrain, Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, EndModel, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData,LSTM_TYPE, DataSet)
cost_seta = BAD.leta()
np.save("./data/bt1/cost_seta.npy",cost_seta)
print("end")

##########################################################
Backday = [vectorwind_grid24[5],vectorprice_grid24[5]]
LSTM_TYPE = "Vector"
WindModel = load_model("./winddata/windowwind24.h5")
PriceModel = load_model("./pricedata/windowprice24.h5")

BAD = BenchmarkInAccurateDemand(Capacity, Ntrain, Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, EndModel, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData, LSTM_TYPE, DataSet)
cost_veta = BAD.leta()
np.save("./data/bt1/cost_veta.npy",cost_veta)
print("end")


cr =5; Capacity = int(max(l)*10000*cr); 
##########################################################
Backday = [end_grid24_5[1],end_grid24_5[1]]
device = torch.device("cpu")
LSTM_TYPE = "End"
EndModel = lstm(13,int(end_grid24_5[2]),24,int(end_grid24_5[3]),int(end_grid24_5[1]))
EndModel.load_state_dict(torch.load("./enddata/endtoend245.pth",device))
EndModel = EndModel.to(device)
WindModel = load_model("./winddata/singlewind.h5")
PriceModel = load_model("./pricedata/singleprice.h5")
BAD = BenchmarkInAccurateDemand(Capacity, Ntrain, Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, EndModel, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData, LSTM_TYPE, DataSet)
cost_eeta = BAD.eeta()
np.save("./data/bt5/cost_eeta.npy",cost_eeta)
print("end")

##########################################################
Backday = [singlewind_grid[5],singleprice_grid[5]]
LSTM_TYPE = "Point"
WindModel = load_model("./winddata/singlewind.h5")
PriceModel = load_model("./pricedata/singleprice.h5")
BAD = BenchmarkInAccurateDemand(Capacity, Ntrain, Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, EndModel, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData, LSTM_TYPE, DataSet)
cost_deta= BAD.deta()
np.save("./data/bt5/cost_deta.npy",cost_deta)
print("end")
cost_mpc= BAD.mpc()
np.save("./data/bt5/cost_mpc.npy",cost_mpc)
print("end")
cost_thb= BAD.thb()
np.save("./data/bt5/cost_thb.npy",cost_thb)
print("end")
cost_ofl = BAD.ofl()
np.save("./data/bt5/cost_ofl.npy",cost_ofl)
print("end")
cost_peta = BAD.leta()
np.save("./data/bt5/cost_peta.npy",cost_peta)
print("end")
cost_nos = BAD.nos()
np.save("./data/bt5/cost_nos.npy",cost_nos)
print("end")

##########################################################
Backday = [s2swind_grid24[5],s2swind_grid24[5]]
LSTM_TYPE = "S2S"
WindModel = Seq2Seq(output_dim=1, hidden_dim=int(s2swind_grid24[0]), output_length=24, input_shape=(Backday[0], dataset_r.shape[1]), peek=False, depth=int(s2swind_grid24[1]), dropout = 0.2)

PriceModel = Seq2Seq(output_dim=1, hidden_dim=int(s2sprice_grid24[0]), output_length=24, input_shape=(Backday[1], dataset_r.shape[1]), peek=False, depth=int(s2sprice_grid24[1]), dropout = 0.2)

WindModel.load_weights("./winddata/s2swind24.h5")
PriceModel.load_weights("./pricedata/s2sprice24.h5")

BAD = BenchmarkInAccurateDemand(Capacity, Ntrain, Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, EndModel, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData,LSTM_TYPE, DataSet)
cost_seta = BAD.leta()
np.save("./data/bt5/cost_seta.npy",cost_seta)
print("end")

##########################################################
Backday = [vectorwind_grid24[5],vectorprice_grid24[5]]
LSTM_TYPE = "Vector"
WindModel = load_model("./winddata/windowwind24.h5")
PriceModel = load_model("./pricedata/windowprice24.h5")

BAD = BenchmarkInAccurateDemand(Capacity, Ntrain, Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, EndModel, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData, LSTM_TYPE, DataSet)
cost_veta = BAD.leta()
np.save("./data/bt5/cost_veta.npy",cost_veta)
print("end")


cr = 0.5; Capacity = int(max(l)*10000*cr); 
##########################################################
Backday = [end_grid24_05[1],end_grid24_05[1]]
device = torch.device("cpu")
LSTM_TYPE = "End"
EndModel = lstm(13,int(end_grid24_05[2]),24,int(end_grid24_05[3]),end_grid24_05[1])
EndModel.load_state_dict(torch.load("./enddata/endtoend240.5.pth",device))
EndModel = EndModel.to(device)
WindModel = load_model("./winddata/singlewind.h5")
PriceModel = load_model("./pricedata/singleprice.h5")
BAD = BenchmarkInAccurateDemand(Capacity, Ntrain, Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, EndModel, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData, LSTM_TYPE, DataSet)
cost_eeta = BAD.eeta()
np.save("./data/bt0/cost_eeta.npy",cost_eeta)
print("end")

##########################################################
Backday = [singlewind_grid[5],singleprice_grid[5]]
LSTM_TYPE = "Point"
WindModel = load_model("./winddata/singlewind.h5")
PriceModel = load_model("./pricedata/singleprice.h5")
BAD = BenchmarkInAccurateDemand(Capacity, Ntrain, Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, EndModel, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData, LSTM_TYPE, DataSet)
cost_deta= BAD.deta()
np.save("./data/bt0/cost_deta.npy",cost_deta)
print("end")
cost_mpc= BAD.mpc()
np.save("./data/bt0/cost_mpc.npy",cost_mpc)
print("end")
cost_thb= BAD.thb()
np.save("./data/bt0/cost_thb.npy",cost_thb)
print("end")
cost_ofl = BAD.ofl()
np.save("./data/bt0/cost_ofl.npy",cost_ofl)
print("end")
cost_peta = BAD.leta()
np.save("./data/bt0/cost_peta.npy",cost_peta)
print("end")
cost_nos = BAD.nos()
np.save("./data/bt0/cost_nos.npy",cost_nos)
print("end")

##########################################################
Backday = [s2swind_grid24[5],s2swind_grid24[5]]
LSTM_TYPE = "S2S"
WindModel = Seq2Seq(output_dim=1, hidden_dim=int(s2swind_grid24[0]), output_length=24, input_shape=(Backday[0], dataset_r.shape[1]), peek=False, depth=int(s2swind_grid24[1]), dropout = 0.2)

PriceModel = Seq2Seq(output_dim=1, hidden_dim=int(s2sprice_grid24[0]), output_length=24, input_shape=(Backday[1], dataset_r.shape[1]), peek=False, depth=int(s2sprice_grid24[1]), dropout = 0.2)

WindModel.load_weights("./winddata/s2swind24.h5")
PriceModel.load_weights("./pricedata/s2sprice24.h5")

BAD = BenchmarkInAccurateDemand(Capacity, Ntrain, Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, EndModel, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData,LSTM_TYPE, DataSet)
cost_seta = BAD.leta()
np.save("./data/bt0/cost_seta.npy",cost_seta)
print("end")

##########################################################
Backday = [vectorwind_grid24[5],vectorprice_grid24[5]]
LSTM_TYPE = "Vector"
WindModel = load_model("./winddata/windowwind24.h5")
PriceModel = load_model("./pricedata/windowprice24.h5")

BAD = BenchmarkInAccurateDemand(Capacity, Ntrain, Ntest,Nvalid, Backday, Interval, Price, Demand, Renewables, EndModel, PriceModel, WindModel,PriceScalar,WindScalar,WindStackData,PriceStackData, LSTM_TYPE, DataSet)
cost_veta = BAD.leta()
np.save("./data/bt0/cost_veta.npy",cost_veta)
print("end")
