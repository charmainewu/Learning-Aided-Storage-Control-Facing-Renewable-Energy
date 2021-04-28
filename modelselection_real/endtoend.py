# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:21:44 2021

@author: jmwu
"""

from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
#from keras.callbacks import EarlyStopping
#from sklearn.model_selection import ParameterGrid
#from sklearn.utils.random import sample_without_replacement
#import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import ParameterGrid
from sklearn.utils.random import sample_without_replacement

use_gpu = torch.cuda.is_available()
print(use_gpu)

class pen_loss(torch.nn.Module):
    def __init__(self):
        super(pen_loss,self).__init__()
        
    def forward(self,out,var_y):
        p = var_y[:,:n_outputs]
        a = var_y[:,n_outputs:]
        
        delta = out

        cost = torch.zeros(1)
        for t in range(24):
            if t == 0:
              #constraints
              
              max_sign_1 = ( torch.sign(torch.zeros(len(var_y)).cuda()-delta[:,t])+1 )/2
              max_value_1 = max_sign_1*torch.abs(torch.zeros(len(var_y)).cuda()-delta[:,t])
              
              if(use_gpu):
                min_sign_1 = ( torch.sign(delta[:,t]-torch.from_numpy(np.ones(len(var_y))*ml).cuda()) + 1)/2
                min_value_1 = min_sign_1*torch.abs(delta[:,t]-torch.from_numpy(np.ones(len(var_y))*ml).cuda())
              else:
                min_sign_1 = ( torch.sign(delta[:,t]-torch.from_numpy(np.ones(len(var_y))*ml)) + 1)/2
                min_value_1 = min_sign_1*torch.abs(delta[:,t]-torch.from_numpy(np.ones(len(var_y))*ml))
              
              x0 = delta[:,t]
              pen = (max_value_1 + min_value_1)*A

              bill = torch.abs(a[:,t]+delta[:,t])*p[:,t]+pen
              
              cost = cost +torch.mean(bill)

            else:
              
              min_sign_0 = (torch.sign(-delta[:,t]-a[:,t])+1)/2
              min_value_0 = min_sign_0*torch.abs(-delta[:,t]-a[:,t])

              max_sign_1 = (torch.sign(-delta[:,t]-x0)+1)/2
              max_value_1 = max_sign_1*torch.abs(-delta[:,t]-x0)

              if(use_gpu):
                min_sign_1 = (torch.sign(x0+delta[:,t]-torch.from_numpy(np.ones(len(var_y))*ml).cuda()) + 1)/2
                min_value_1 = min_sign_1*torch.abs(x0+delta[:,t]-torch.from_numpy(np.ones(len(var_y))*ml).cuda())
              else:
                min_sign_1 = (torch.sign(x0+delta[:,t]-torch.from_numpy(np.ones(len(var_y))*ml)) + 1)/2
                min_value_1 = min_sign_1*torch.abs(x0+delta[:,t]-torch.from_numpy(np.ones(len(var_y))*ml))
              
              x0 = x0 + delta[:,t]
              
              pen = (min_value_0 + max_value_1 + min_value_1)*A
              
              bill = torch.abs(a[:,t]+delta[:,t])*p[:,t]+pen
              cost = cost +torch.mean(bill)
        return cost


class lstm(nn.Module):
    def __init__(self,input_size=12,hidden_size=4,output_size=1,num_layer=1,seq_length=1):
        super(lstm,self).__init__()
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layer,dropout=drop_sample)
        self.layer2 = nn.Linear(hidden_size*seq_length,output_size)
        
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = torch.swapaxes(x,0,1)
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

scaler_l = MinMaxScaler(feature_range=(0, 1))
l = scaler_l.fit_transform(net_data)
l = scaler_l.inverse_transform(l)/10000

scaler_p = MinMaxScaler(feature_range=(0, 1))
p = scaler_p.fit_transform(price_data)
p = scaler_p.inverse_transform(l)/100

enc_h = OneHotEncoder(handle_unknown='ignore')
h = enc_h.fit_transform(hour_data).toarray()

enc_m = OneHotEncoder(handle_unknown='ignore')
m = enc_m.fit_transform(month_data).toarray()

enc_w = OneHotEncoder(handle_unknown='ignore')
w = enc_w.fit_transform(week_data).toarray()


dataset  = np.hstack((l,p,w,h))
train_size = int(len(dataset) * 0.50)
valid_size = int(len(dataset) * 0.30)
test_size = int(len(dataset) * 0.20)

train, valid, test = dataset[0:train_size,:], dataset[train_size:train_size+valid_size,:], dataset[train_size+valid_size:train_size+valid_size+test_size,:]

A = 10; 
n_iter = 5; n_inputs = 13; 

epochs = [1000,5000]
backday = [6,12,24]
neurons = [216,512]
drop = [0.2]
lr = [1e-3,1e-4,1e-5]
layer = [2,3]

param_grid = dict(epochs=epochs, backday=backday, neurons=neurons, drop=drop, lr = lr, layer=layer)
param_com = list(ParameterGrid(param_grid))


for n_outputs in [6,12,24]:
  for cr in [0.5,1,5]:
    ml = cr*np.max(l); 
    result = {}; count = 0; acc_best = 100000
    
    for i in sample_without_replacement(len(param_com), n_iter, random_state = 5):
        epochs_sample = param_com[i]['epochs']
        backday_sample = param_com[i]['backday']
        neurons_sample = param_com[i]['neurons']
        lr_sample = param_com[i]['lr']
        drop_sample = param_com[i]['drop']
        layer_sample = param_com[i]['layer']
      
        x_train, y_train = [], []
        x_valid, y_valid = [], []
        x_test, y_test = [], []
        
        for i in range(backday_sample,train_size-n_outputs):
          x_train.append(train[i-backday_sample:i,:])
          y_price = np.array(p[i:i+n_outputs]).reshape(1,-1)[0]
          y_demand = np.array(l[i:i+n_outputs]).reshape(1,-1)[0]
          y_train.append(list(y_price)+list(y_demand)) 
        x_train, y_train = np.array(x_train), np.array(y_train)
        #y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
        
        for i in range(backday_sample,valid_size-n_outputs):
          x_valid.append(valid[i-backday_sample:i,:])
          y_price = np.array(p[train_size+i:train_size+i+n_outputs]).reshape(1,-1)[0]
          y_demand = np.array(l[train_size+i:train_size+i+n_outputs]).reshape(1,-1)[0]
          y_valid.append(list(y_price)+list(y_demand))
        x_valid, y_valid = np.array(x_valid), np.array(y_valid)
        #y_valid = y_valid.reshape((y_valid.shape[0], y_valid.shape[1], 1))
        
        for i in range(backday_sample,test_size-n_outputs):
          x_test.append(test[i-backday_sample:i,:])
          y_price = np.array(p[valid_size+train_size+i:valid_size+train_size+i+n_outputs]).reshape(1,-1)[0]
          y_demand = np.array(l[valid_size+train_size+i:valid_size+train_size+i+n_outputs]).reshape(1,-1)[0]
          y_test.append(list(y_price)+list(y_demand))
        x_test, y_test = np.array(x_test), np.array(y_test)
        #y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
        
        train_X = np.swapaxes(x_train,0,1)
        train_Y = y_train
        test_X = np.swapaxes(x_valid,0,1)
        test_Y = y_valid
        
        train_x = torch.from_numpy(train_X).to(torch.float32)
        train_y = torch.from_numpy(train_Y).to(torch.float32)
        test_x = torch.from_numpy(test_X).to(torch.float32)
        test_y = torch.from_numpy(test_Y).to(torch.float32)
        
        model = lstm(n_inputs,neurons_sample,n_outputs,layer_sample,backday_sample)
        criterion = pen_loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_sample)
        
        if(use_gpu):
          model = model.cuda()
          criterion = criterion.cuda()
  
        for e in range(epochs_sample):
            var_x = Variable(train_x)
            var_y = Variable(train_y)
            
            if (use_gpu):
              var_x,var_y = var_x.cuda(),var_y.cuda()
            
            out = model(var_x)
            loss = criterion(out,var_y)
            optimizer.zero_grad()
            loss.backward()
            #for name, parms in model.named_parameters():	
              #print('-->name:', name, '-->grad_requirs:',parms.requires_grad,'-->grad_value:',parms.grad)
            optimizer.step()
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
            if (e + 1) % 100 == 0: # 每 100 次输出结果
                print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
        
        model = model.eval() # 转换成测试模式
        val_x = Variable(test_x)
        val_y = Variable(test_y)
        
        if (use_gpu):
              var_x,var_y = var_x.cuda(),var_y.cuda()
        
        val_out = model(val_x) #测试集的预测结果
        loss = criterion(val_out,val_y)
        
        if(use_gpu):
          loss = loss.cpu()
        
        result[count] = {}
        result[count]["acc"],result[count]["par"] = loss.detach().numpy()[0],[epochs_sample,backday_sample,neurons_sample,layer_sample,drop_sample,lr_sample]
        if result[count]["acc"] < acc_best:
          best = result[count]["par"]
          acc_best = result[count]["acc"]
          torch.save(model.state_dict(), './data/endtoend'+str(n_outputs)+str(cr)+'.pth')
        count = count + 1 
    np.save('./data/endlearning_grid'+str(n_outputs)+str(cr)+'.npy', result) # 注意带上后缀名
