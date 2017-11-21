#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:54:45 2017

@author: asejouk
"""

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
#from dnn_app_utils_v2 import

# Call support Function

def L_layer_model_with_adam(X,Y,layer_dims,learning_rate,mini_batch_size,number_iter,print_cost):
    
    costs=[]
    parameters=initialize_parameters_deep(layer_dims)
    
    v,s = initialize_adam(parameters)
    t=0
    seed=10
    
    # loop gradient descent
    
    for i in range(number_iter):
        
        seed=seed+1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        
        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            
            # Forwards Propagation 
            AL, caches= L_model_forward(minibatch_X,parameters)
        
            # Cost Function 
            cost= compute_cost(AL,minibatch_Y)
        
            # Backward Propagation
            grads=L_model_backward(AL,minibatch_Y,caches)
        
            # Update Parameters
        
            t = t + 1 # Adam counter
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,t, learning_rate)
        
        
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0: # print_cost(true) & i/100 with no leftover 
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per 100)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
