# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 15:42:29 2021

@author: jmwu
"""


import numpy as np
from RL_brain import QLearningTable
import warnings
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping
import keras.backend as K
import keras.initializers as KI
import tensorflow as tf
from keras import regularizers
warnings.filterwarnings("ignore")

class BenchmarkInAccurateDemand:
    def __init__(self, Capacity,Window, Ntrain,Ntest,Nvalid, Price,Demand, Renewables,learning_rate,reward_decay,e_greedy,vanish,epochs):
      self.B = Capacity
      self.W = Window
      self.P = Price
      self.D = Demand
      self.R = Renewables
      self.N = Ntrain
      self.T = Ntest
      self.V = Nvalid
      self.S = Ntest
      self.learning_rate = learning_rate
      self.reward_decay = reward_decay
      self.e_greedy = e_greedy
      self.vanish = vanish
      self.epochs = epochs

      
    def stepto(self, action,observation,step,p,pbar,d):
        length = len(p)
        pbar = (1-self.vanish)*pbar+self.vanish*observation[0]
        if action == 0:
            reward = 0
            sl = observation[2]
            ch = 0
            dh = 0
            gh = observation[1]
            stepcost = observation[0]*observation[1]
        elif action == 1:
            reward = (observation[0]-pbar)*min(observation[2],observation[1])
            sl = observation[2]-min(observation[2],observation[1])
            ch = 0
            dh = min(observation[2],observation[1])
            gh = observation[1]-min(observation[2],observation[1])
            stepcost = observation[0]*(observation[1]-min(observation[2],observation[1]))
        elif action == 2:
            reward = (pbar-observation[0])*int((self.B-observation[2]))
            sl = observation[2]+int((self.B-observation[2]))
            ch = int((self.B-observation[2]))
            dh = 0
            gh = observation[1]
            stepcost = observation[0]*(observation[1]+int((self.B-observation[2])))
        
        if step == length-1:
            done = True
            observation_ =  'terminal'
            return observation_, reward, done, pbar, stepcost, sl, ch, dh, gh
        else:
            done = False
            observation_ =  np.array([p[step+1],d[step+1],sl])
            return observation_, reward, done, pbar, stepcost, sl, ch, dh, gh

    def train(self, d,p,RL):
        d = np.array(d)
        d = d.reshape(self.N)
        p = np.array(p)
        p = p.reshape(self.N)
        level = np.array([10,10000,10000])
        for episode in range(self.epoch):
            s = 0; step = 0; pbar = p[0]
            observation = np.array([p[0],d[0],s])
            while True:
                temp_ob = observation.copy()/level; temp_ob = temp_ob.astype(int);
                action = RL.choose_action(str(temp_ob))
                observation_, reward, done, pbar, stepcost,sl, ch, dh, gh = self.stepto(action,observation,step,p,pbar,d)
                if observation_== 'terminal':
                    RL.learn(str(temp_ob), action, reward, observation_)
                    print('terminal episode '+str(episode))
                    break
                else:
                    temp_ob_ = observation_.copy()/level; temp_ob_ = temp_ob_.astype(int);
                    RL.learn(str(temp_ob), action, reward, str(temp_ob_))
                observation = observation_
                step = step + 1
                if step>=(len(d)):
                    break
        return RL
    
    def rl(self):
        RL = QLearningTable(actions=list(range(3)),learning_rate = self.learning_rate, reward_decay = self.reward_decay, e_greedy = self.e_greedy)
        RL = self.train(self.D[:self.N],self.P[:self.N],RL)
        level = np.array([10,10000,10000])
        n_interval = int(self.T/self.I); 
        cost_rl = np.zeros(n_interval)
        for n in range(1, n_interval+1):
            a_real = self.D[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            r_real = self.R[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            p_real = self.P[self.N+self.V+(n-1)*self.I:self.N+self.V+n*self.I]
            d_real = (a_real - r_real)
            
            d_real = d_real.astype(int)
            
            d_real = d_real.reshape(len(d_real))
            p_real = p_real.reshape(len(p_real))
            
            s = 0; step = 0;pbar = p_real[0]
            observation = np.array([p_real[0],d_real[0],s])
            while True:
                temp_ob = observation.copy()/level; temp_ob = temp_ob.astype(int);
                action = RL.choose_action(str(temp_ob))
                observation_, reward, done, pbar, stepcost,sl, cd, dd, gd = self.stepto(action,observation,step,p_real,pbar,d_real)
                if observation_== 'terminal':
                    cost_rl[n-1] = cost_rl[n-1] + stepcost
                    break
                else:
                    cost_rl[n-1] = cost_rl[n-1] + stepcost
                observation = observation_
                step = step + 1
                if step>=self.I:
                    break
        cost_rl_copy = cost_rl.copy()
        for i in range(len(cost_rl_copy)):
          cost_rl[i] = sum(cost_rl_copy[:i+1])
        return cost_rl
      
    def Aour_demand(self,delta,a,p,xc):
        T = int(len(delta)/2); cost = 0
        x = np.zeros(T)
        
        for t in range(T):
            if xc < a[t]:
              a[t] = a[t] - xc
              xc = 0
              break
            else:
              a[t] = 0
              xc = xc - a[t]
              
        for t in range(T):
            if t == 0 :
                x[t] = min(max(0.0,-min(delta[t],a[t])),self.B)
                vb = max(x[t]-0,0)
                va = a[t]-min(x[t]-0,0)
                cost = cost + (va+vb)*p[t]
            else:
                x[t] = min(max(0.0,x[t-1]-min(delta[t],a[t])),self.B)
                vb = max(x[t]-x[t-1],0)
                va = a[t]-min(x[t]-x[t-1],0)
                cost = cost + (va+vb)*p[t]
        return cost,x[t]

    def isDecompose(self,at,x0):
        Acc_sum = 0
        Acc = np.zeros(len(at))
        for i in range(len(at)):
            Acc[i] = Acc_sum + at[i]
            Acc_sum = Acc_sum + at[i]
        AccB = Acc + self.B
        
        sadwAB = np.zeros(int(max(Acc)+1))
        
        for i in range(len(at)):
            if AccB[i] <= max(Acc):
                sadwAB[int(AccB[i])] = 1
            sadwAB[int(Acc[i])] = 1
        
        sadwA = np.zeros(int(max(Acc)+1))
        sadwB = np.zeros(int(max(Acc)+1))
        
        i = 0; j = int(Acc[0]); 
        while(j <= max(Acc) and i+1<=len(at)-1):
            while(i+1<=len(at)-1 and Acc[i+1]==Acc[i]):
                try:
                    sadwA[j] = i+1
                    i = i + 1
                except:
                    break
            while(i+1<=len(at)-1 and Acc[i+1]>Acc[i]):
                k = Acc[i+1]-Acc[i]
                try:
                    sadwA[j] = i + 1
                except: 
                    break
                while(k > 0):
                    j = j + 1
                    try:
                        sadwA[j] = i + 1
                        k = k - 1
                    except:
                        break
                i = i + 1
                
        i = 0; j = int(AccB[0]); 
        while(j <= max(Acc) and i+1<=len(at)-1):
            while(i+1<=len(at)-1 and AccB[i+1]==AccB[i]):
                i = i + 1
            while(i+1<=len(at)-1 and AccB[i+1]>AccB[i]):
                k = AccB[i+1]-AccB[i]
                while(k > 0):
                    j = j + 1
                    try:
                        sadwB[j] = i + 1
                        k = k - 1
                    except:
                        break
                i = i + 1
        
        a_index =np.where(sadwAB==1)[0]
        
        a = [];ts = [];tnz = [];
        for i in range(len(a_index)-1):
            a.append(a_index[i+1]-a_index[i])
            ts.append(sadwB[a_index[i+1]])
            tnz.append(sadwA[a_index[i]])
        
        Trunc_sum = a[0]; t = 0; del_list = [];
        while(Trunc_sum <= x0):
            del_list.append(t)
            t = t + 1
            try:
                Trunc_sum = Trunc_sum + a[t]
            except:
                break
                    
        for i in del_list:
            del a[i]
            del ts[i]
            del tnz[i]
        a[0] = Trunc_sum - x0
        return a, ts, tnz


    def Aofl(self,ts,tnz,abar,p):
        mu_c = 1000000; mu_d =1000000
        a = np.zeros(len(p))
        a[int(tnz)] = abar
        xt = np.zeros(len(p))
        dt = np.zeros(len(p))
        vat = np.zeros(len(p))
        vbt = np.zeros(len(p))
        try:
            p_min = min(p[int(ts):int(tnz+1)])
        except:
            p_min = 1e8
                
        if p[int(ts)]==p_min:
            dt[int(ts)] = 0
            vat[int(ts)] = a[int(ts)]
            vbt[int(ts)] = min(max(abar-0,0),mu_c)
            xt[int(ts)] = 0 + vbt[int(ts)] - dt[int(ts)]
        else:
            dt[int(ts)] = min(a[int(ts)],mu_d,0)
            vat[int(ts)] = a[int(ts)]-dt[int(ts)]
            vbt[int(ts)] = 0
            xt[int(ts)] = 0 + vbt[int(ts)] - dt[int(ts)]
        
        for t in range(int(ts+1),int(tnz)):
            if p[t] == p_min:
                dt[t] = 0
                vat[t] = a[t]
                vbt[t] = min(max(abar-xt[t-1],0),mu_c)
                xt[t] = xt[t-1] + vbt[t] - dt[t]
            else:
                dt[t] = min(a[t],mu_d,xt[t-1])
                vat[t] = a[t]-dt[t]
                vbt[t] = 0
                xt[t] = xt[t-1] + vbt[t] - dt[t]
                
        dt[int(tnz)] = min(a[int(tnz)],mu_d,xt[int(tnz-1)])
        vat[int(tnz)] = a[int(tnz)]-dt[int(tnz)]
        vbt[int(tnz)] = 0
        xt[int(tnz)] = xt[int(tnz-1)] + vbt[int(tnz)] - dt[int(tnz)]
        
        return xt,dt,vat,vbt
    
    def Aofl_hat(self,ao,po,xc):
        a = ao.copy() ; p = po.copy()
        x = np.zeros(int(len(a)))
        d = np.zeros(int(len(a)))
        va = np.zeros(int(len(a)))
        vb = np.zeros(int(len(a)))
        x0 = xc; x[0] = 0
        abar, ts, tnz = self.isDecompose(a,x0)
        cost_shot = np.zeros(len(a))
        for i in range(len(abar)):
            xt,dt,vat,vbt = self.Aofl(ts[i],tnz[i],abar[i],p)
            for t in range(int(ts[i]),int(tnz[i]+1)):
                x[t] = x[t] + xt[t]
                d[t] = d[t] + dt[t]
                va[t] = va[t] + vat[t]
                vb[t] = vb[t] + vbt[t]
                a[t] = a[t] - dt[t] - vat[t]
                if vat[t]+vbt[t]>0:
                    cost_shot[t] = cost_shot[t] + p[t]*(vat[t]+vbt[t])
        return x,d,va,vb,cost_shot
    
    def ofl(self):
        n_interval = int(self.T/self.I)
        cost_ofl = np.zeros(n_interval)
        for n in range(1, n_interval+1):
            a_real = self.D[self.N+self.V:self.N+self.V+n*self.I]
            r_real = self.R[self.N+self.V:self.N+self.V+n*self.I]
            p_real = self.P[self.N+self.V:self.N+self.V+n*self.I]
            d_real = (a_real - r_real)
            
            d_real = d_real.astype(int); d_real = np.r_[np.array(0),d_real]
            p_real = np.r_[np.array(self.P[self.N+self.V-1]),p_real]
            
            x,d,va,vb,cost_shot = self.Aofl_hat(d_real,p_real,0)
            cost_ofl[n-1] = sum(cost_shot)
        return cost_ofl