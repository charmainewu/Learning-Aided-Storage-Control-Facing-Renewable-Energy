# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 23:49:48 2021

@author: jmwu
"""


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
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import load_model
from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
import matplotlib.pyplot as plt

###############################################################################
deta = np.load("./data/bt0/cost_deta.npy")
ofl = np.load("./data/bt0/cost_ofl.npy")
nos = np.load("./data/bt0/cost_nos.npy")
mpc = np.load("./data/bt0/cost_mpc.npy")
thb = np.load("./data/bt0/cost_thb.npy")
s2slstm = np.load("./data/bt0/cost_seta.npy")
vectorlstm = np.load("./data/bt0/cost_veta.npy")
pointlstm = np.load("./data/bt0/cost_peta.npy")
endlstm = np.load("./data/bt0/cost_eeta.npy")

x = list(range(24,24*12*3,24*3))
    
fig, axs = plt.subplots(figsize=(6, 4),constrained_layout=True)
#axs.plot(x, nos[0:36:3]/ofl[0:36:3],  linewidth=2,c = 'royalblue',marker = "o", markersize = 10, label='No Storage')
#axs.plot(x, rl[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "p",  markersize = 12,c='lightseagreen',label='RL')
#axs.plot(x, mpc[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "h",  markersize = 12,c='gold',label='MPC')
#axs.plot(x, thb[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "<",  markersize = 12,c='royalblue',label='THB')
axs.plot(x, deta[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "*",  markersize = 12,c='wheat',label='DETA')
axs.plot(x, s2slstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = ">", markersize = 10,c='orange',label='SPTA')
axs.plot(x, vectorlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "s", markersize = 10,c='salmon', label='VPTA')
axs.plot(x, pointlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "p", markersize = 10,c='yellowgreen', label='PPTA')
axs.plot(x, endlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "o", markersize = 10,c='lightseagreen', label='EETA')
axs.plot(x, nos[0:36:3]/ofl[0:36:3], linewidth=2, marker = "v", markersize = 10,c='slateblue', label='NOS')


plt.xlabel('Time/h', fontsize = 20)
plt.ylabel(r'$ \alpha $', fontsize = 20)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
axs.legend(fontsize = 14,loc=1)
plt.savefig("./figure/inaccur1b.pdf")


###############################################################################
deta = np.load("./data/bt1/cost_deta.npy")
ofl = np.load("./data/bt1/cost_ofl.npy")
nos = np.load("./data/bt1/cost_nos.npy")
mpc = np.load("./data/bt1/cost_mpc.npy")
thb = np.load("./data/bt1/cost_thb.npy")
s2slstm = np.load("./data/bt1/cost_seta.npy")
vectorlstm = np.load("./data/bt1/cost_veta.npy")
pointlstm = np.load("./data/bt1/cost_peta.npy")
endlstm = np.load("./data/bt1/cost_eeta.npy")

x = list(range(24,24*12*3,24*3))
    
fig, axs = plt.subplots(figsize=(6, 4),constrained_layout=True)
#axs.plot(x, nos[0:36:3]/ofl[0:36:3],  linewidth=2,c = 'royalblue',marker = "o", markersize = 10, label='No Storage')
#axs.plot(x, rl[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "p",  markersize = 12,c='lightseagreen',label='RL')
#axs.plot(x, mpc[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "h",  markersize = 12,c='gold',label='MPC')
#axs.plot(x, thb[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "<",  markersize = 12,c='royalblue',label='THB')
axs.plot(x, deta[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "*",  markersize = 12,c='wheat',label='DETA')
axs.plot(x, s2slstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = ">", markersize = 10,c='orange',label='SPTA')
axs.plot(x, vectorlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "s", markersize = 10,c='salmon', label='VPTA')
axs.plot(x, pointlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "p", markersize = 10,c='yellowgreen', label='PPTA')
axs.plot(x, endlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "o", markersize = 10,c='lightseagreen', label='EETA')
axs.plot(x, nos[0:36:3]/ofl[0:36:3], linewidth=2, marker = "v", markersize = 10,c='slateblue', label='NOS')

plt.xlabel('Time/h', fontsize = 20)
plt.ylabel(r'$ \alpha $', fontsize = 20)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
#axs.legend(fontsize = 14,loc=3)
plt.savefig("./figure/inaccur2b.pdf")

###############################################################################
deta = np.load("./data/bt5/cost_deta.npy")
ofl = np.load("./data/bt5/cost_ofl.npy")
nos = np.load("./data/bt5/cost_nos.npy")
mpc = np.load("./data/bt5/cost_mpc.npy")
thb = np.load("./data/bt5/cost_thb.npy")
s2slstm = np.load("./data/bt5/cost_seta.npy")
vectorlstm = np.load("./data/bt5/cost_veta.npy")
pointlstm = np.load("./data/bt5/cost_peta.npy")
endlstm = np.load("./data/bt5/cost_eeta.npy")

x = list(range(24,24*12*3,24*3))
    
fig, axs = plt.subplots(figsize=(6, 4),constrained_layout=True)
#axs.plot(x, nos[0:36:3]/ofl[0:36:3],  linewidth=2,c = 'royalblue',marker = "o", markersize = 10, label='No Storage')
#axs.plot(x, rl[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "p",  markersize = 12,c='lightseagreen',label='RL')
#axs.plot(x, mpc[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "h",  markersize = 12,c='gold',label='MPC')
#axs.plot(x, thb[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "<",  markersize = 12,c='royalblue',label='THB')
axs.plot(x, deta[0:36:3]/ofl[0:36:3],  linewidth=2,marker = "*",  markersize = 12,c='wheat',label='DETA')
axs.plot(x, s2slstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = ">", markersize = 10,c='orange',label='SPTA')
axs.plot(x, vectorlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "s", markersize = 10,c='salmon', label='VPTA')
axs.plot(x, pointlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "p", markersize = 10,c='yellowgreen', label='PPTA')
axs.plot(x, endlstm[0:36:3]/ofl[0:36:3], linewidth=2, marker = "o", markersize = 10,c='lightseagreen', label='EETA')
axs.plot(x, nos[0:36:3]/ofl[0:36:3], linewidth=2, marker = "v", markersize = 10,c='slateblue', label='NOS')

plt.xlabel('Time/h', fontsize = 20)
plt.ylabel(r'$ \alpha $', fontsize = 20)
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
#axs.legend()
plt.savefig("./figure/inaccur3b.pdf")