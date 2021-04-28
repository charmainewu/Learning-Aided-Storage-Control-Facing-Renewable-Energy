# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 20:47:12 2021

@author: jmwu
"""

import numpy as np

singleprice_grid = np.load('./data/singleprice_best1.npy',allow_pickle=True) 
singlewind_grid = np.load('./data/singlewind_best1.npy',allow_pickle=True) 

s2sprice_grid6 = np.load('./data/s2sprice_best6.npy',allow_pickle=True) 
s2swind_grid6 = np.load('./data/s2swind_best6.npy',allow_pickle=True) 
s2sprice_grid12 = np.load('./data/s2sprice_best12.npy',allow_pickle=True) 
s2swind_grid12 = np.load('./data/s2swind_best12.npy',allow_pickle=True) 
s2sprice_grid24 = np.load('./data/s2sprice_best24.npy',allow_pickle=True) 
s2swind_grid24 = np.load('./data/s2swind_best24.npy',allow_pickle=True)
 
vectorprice_grid6 = np.load('./data/windowprice_best6.npy',allow_pickle=True) 
vectorwind_grid6 = np.load('./data/windowwind_best6.npy',allow_pickle=True) 
vectorprice_grid12 = np.load('./data/windowprice_best12.npy',allow_pickle=True) 
vectorwind_grid12 = np.load('./data/windowwind_best12.npy',allow_pickle=True) 
vectorprice_grid24 = np.load('./data/windowprice_best24.npy',allow_pickle=True) 
vectorwind_grid24 = np.load('./data/windowwind_best24.npy',allow_pickle=True) 



