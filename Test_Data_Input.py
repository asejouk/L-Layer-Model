#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:05:54 2017

@author: asejouk
"""
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy 
# Load data

d=scipy.io.loadmat('data2.mat')

train_x=d["X"].T
train_y=d["y"].T
test_x=d["Xval"].T
test_y=d["yval"].T

layers_dims=[train_x.shape[0],5,2,1]
lambd=0.7
learning_rate=0.0007
number_iterations=50000
keep_prob=0.5
mini_batch_size=256



#parameters= L_layer_model_with_adam(train_x,train_y,layers_dims,learning_rate,mini_batch_size,number_iterations,print_cost=True)

#prediction, accuracy =predict(train_x,train_y,parameters)
#print(accuracy)
#prediction, accuracy =predict(test_x,test_y,parameters)
#print(accuracy)

parameters= L_layer_model(train_x,train_y,layers_dims,learning_rate,number_iterations,print_cost=True)

prediction, accuracy =predict(train_x,train_y,parameters)
print(accuracy)
prediction, accuracy =predict(test_x,test_y,parameters)
print(accuracy)