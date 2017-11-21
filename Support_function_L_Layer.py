#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 17:10:45 2017

@author: asejouk
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import math
from pdb import set_trace as bp
np.random.seed(1)

def initialize_parameters_deep(layer_dims):
    # This uses xavier initilizer, objective is to keep var(input) and var(output) the same sq(2/n(input layer))
    parameters={}
    L=len(layer_dims)
    np.random.seed(3)
    
    for i in range(1,L):
        parameters["W"+str(i)]= np.random.randn(layer_dims[i],layer_dims[i-1])*np.sqrt(2/layer_dims[i-1])
        parameters["b"+str(i)]= np.zeros((layer_dims[i],1))
        
    return parameters

def linear_forward(A,W,b):
    
    Z=np.dot(W,A)+b
    
    cache=(A,W,b)
    return Z, cache

def sigmoid(Z):
    A=1/(1+np.exp(-Z))
    return A,Z

def relu(Z):
    A=np.maximum(0,Z)
    return A,Z

def softmax(Z):
    
    A=np.exp(Z)/np.sum(np.exp(Z),axis=0).reshape((1,Z.shape[1]))
    
    return A,Z
    
    
def linear_activation_forward(A_prev,W,b,activation):
    
    if activation == "sigmoid":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)
    
    elif activation == "relu":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=relu(Z)
        
    elif activation == "softmax":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=softmax(Z)
        
    cache=(linear_cache,activation_cache)
    return A,cache
    
    
def L_model_forward(X,parameters):
    L=len(parameters)//2
    caches=[]
    A=X
    for i in range(1,L):
        A_prev=A
        A, cache=linear_activation_forward(A_prev,parameters["W"+str(i)],parameters["b"+str(i)],"relu")
        caches.append(cache)
    
    # The last layer is sigmoid the for loop goes from 1 to (L-1)
    
    AL, cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"softmax")
    caches.append(cache)
    return AL,caches

def L_model_forward_with_dropout(X,parameters,keep_prob):
    #np.random.seed(1)
    L=len(parameters)//2
    caches=[]
    A=X
    for i in range(1,L):
        A_prev=A
        A, cache=linear_activation_forward(A_prev,parameters["W"+str(i)],parameters["b"+str(i)],"relu")
        # Inverted Dropout code
        D=np.random.rand(A.shape[0],A.shape[1])
        D=(D<keep_prob)
        A=(D*A)/keep_prob  # zero some of the activations in current layer and equlize by dividing by keep_prob
        caches.append((cache,D))
     
    # The last layer is sigmoid the for loop goes from 1 to (L-1)

    AL, cache=linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"softmax")
    caches.append(cache)
    return AL,caches

def compute_cost(AL,Y):
    
    m=Y.shape[1]
    
    cost = np.sum(np.diagonal(np.matmul(Y.T,np.log(AL))))/-m
    return cost


def compute_cost_with_regularization(AL,Y,parameters,lambd):
    
    m=Y.shape[1]
    L=len(parameters)//2
    L2_regularization_cost=0
    
    cross_entropy_cost = np.sum(np.diagonal(np.matmul(Y.T,np.log(AL))))/-m
    
    for i in range(1,L+1):
        
        L2_regularization_cost += 0.5*lambd*np.sum(np.square(parameters["W"+str(i)]))/m
    
    cost=cross_entropy_cost+L2_regularization_cost
    
    return cost

def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev, dW, db


def relu_backward(dA,activation_cache):
    drelu=1*(activation_cache>0)
    dZ=np.multiply(drelu,dA)
    return dZ

def sigmoid_backward(dA, activation_cache):
    A,temp22 =sigmoid(activation_cache)
    dZ=np.multiply(np.multiply(A,(1-A)),dA)
    return dZ
    

def linear_activation_backward(dA,cache,activation):
    
    linear_cache, activation_cache=cache
    if activation=="relu":
        dZ=relu_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
        
    elif activation=="sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)  
        
    elif activation=="softmax":
        dZ=dA
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL,Y,caches):
    
    grads={}
    grads1={}
    L=len(caches)
    Y = Y.reshape(AL.shape)
    dAL=np.divide(Y,AL)
    dZ=AL-Y                         # We calculate directly dZ here instade of passing dAL in the linear_activation_backward function. 
    grads["dA"+str(L)]=dAL
    current_cache=caches[L-1]
    dA_prev, dW_temp, db_temp=linear_activation_backward(dZ,current_cache,"softmax") # we pass dZ only for softmax option as output
    grads["dA"+str(L-1)]=dA_prev
    grads["dW"+str(L)]=dW_temp
    grads["db"+str(L)]=db_temp
    # grads1 is used for gradient check 
    grads1["W"+str(L)]=dW_temp
    grads1["b"+str(L)]=db_temp
    
    for i in reversed(range(L-1)):
        current_cache=caches[i]
        dA_prev_temp, dW_temp, db_temp =linear_activation_backward(grads["dA"+str(i+1)],current_cache,"relu")
        grads["dA"+str(i)]=dA_prev_temp
        grads["dW"+str(i+1)]=dW_temp
        grads["db"+str(i+1)]=db_temp
        # grads1 is used for gradient check 
        grads1["W"+str(i+1)]=dW_temp
        grads1["b"+str(i+1)]=db_temp
        
    
    
    return grads

def L_model_backward_with_dropout(AL,Y,caches,keep_prob):
    
    grads={}
    L=len(caches) # For L=3 
    Y = Y.reshape(AL.shape)
    dAL=np.divide(Y,AL)
    dZ=AL-Y
    grads["dA"+str(L)]=dAL # Than this is dA3
    current_cache=caches[L-1]# This index into 2 (starts from 0
    dA_prev, dW_temp, db_temp=linear_activation_backward(dZ,current_cache,"softmax") 
    grads["dA"+str(L-1)]=dA_prev # This is than dA2
    grads["dW"+str(L)]=dW_temp # This is dW3
    grads["db"+str(L)]=db_temp
    for i in reversed(range(L-1)):
        
        current_cache,D = caches[i]
        grads["dA"+str(i+1)]=(grads["dA"+str(i+1)]*D)/keep_prob # first loop: D2 and dA2, second loop D1 dA1
        dA_prev, dW_temp, db_temp =linear_activation_backward(grads["dA"+str(i+1)],current_cache,"relu")
        grads["dA"+str(i)]=dA_prev # first loop A1, second loop A0
        grads["dW"+str(i+1)]=dW_temp # first loop dW2, second loop dW1
        grads["db"+str(i+1)]=db_temp
        
    
    return grads

def L_model_backward_with_regularization(AL,Y,caches,lambd):
    
    grads={}
    L=len(caches)
    Y = Y.reshape(AL.shape)
    dAL=np.divide(Y,AL)
    dZ=AL-Y
    grads["dA"+str(L)]=dAL
    current_cache=caches[L-1]
    # Regularization input
    temp_1,temp_2=current_cache
    A_temp,W_temp,b_temp=temp_1
    m=Y.shape[1]
    dA_prev, dW_temp, db_temp=linear_activation_backward(dZ,current_cache,"softmax")
    grads["dA"+str(L-1)]=dA_prev
    grads["dW"+str(L)]=dW_temp+(lambd*W_temp/m)
    grads["db"+str(L)]=db_temp
    
    for i in reversed(range(L-1)):
        current_cache=caches[i]
        temp_1,temp_2=current_cache
        A_temp,W_temp,b_temp=temp_1
        dA_prev_temp, dW_temp, db_temp =linear_activation_backward(grads["dA"+str(i+1)],current_cache,"relu")
        grads["dA"+str(i)]=dA_prev_temp
        grads["dW"+str(i+1)]=dW_temp+lambd*W_temp/m
        grads["db"+str(i+1)]=db_temp
    
    
    return grads
    
def update_parameters(parameters,grads,learning_rate):
    L=len(parameters)//2
    for i in range(1,L+1):
        parameters["W"+str(i)]=parameters["W"+str(i)]-(learning_rate*grads["dW"+str(i)])
        parameters["b"+str(i)]=parameters["b"+str(i)]-(learning_rate*grads["db"+str(i)])
        
    return parameters



def initialize_adam(parameters):
    
    L=len(parameters)//2
    v={}
    s={}
    
    for i in range(L):
        
        # initialize momentum v dictionary
        v["dW" + str(i+1)] = np.zeros((parameters["W"+str(i+1)].shape[0],parameters["W"+str(i+1)].shape[1]))
        v["db"+str(i+1)]=np.zeros((parameters["b"+str(i+1)].shape[0],parameters["b"+str(i+1)].shape[1]))
        
        # initialize RMS Prop s dictionary 
        s["dW"+str(i+1)]=np.zeros((parameters["W"+str(i+1)].shape[0],parameters["W"+str(i+1)].shape[1]))
        s["db"+str(i+1)]=np.zeros((parameters["b"+str(i+1)].shape[0],parameters["b"+str(i+1)].shape[1]))
        
    return v,s

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
    
    
    L=len(parameters)//2    # number of layers 
    v_corrected={}
    s_corrected={}
    for i in range(L):
        # First we update v dictionary 
        v["dW"+str(i+1)]=beta1*v["dW"+str(i+1)]+(1-beta1)*grads["dW"+str(i+1)]
        v["db"+str(i+1)]=beta1*v["db"+str(i+1)]+(1-beta1)*grads["db"+str(i+1)]
        
        # bias correction 
        v_corrected["dW"+str(i+1)]=v["dW"+str(i+1)]/(1-np.power(beta1,t))
        v_corrected["db"+str(i+1)]=v["db"+str(i+1)]/(1-np.power(beta1,t))
        
        # Update s dictionary 
        s["dW"+str(i+1)]=beta2*s["dW"+str(i+1)]+(1-beta2)*np.square(grads["dW"+str(i+1)])
        s["db"+str(i+1)]=beta2*s["db"+str(i+1)]+(1-beta2)*np.square(grads["db"+str(i+1)])
        
        # bias correction 
        s_corrected["dW"+str(i+1)]=s["dW"+str(i+1)]/(1-np.power(beta2,t))
        s_corrected["db"+str(i+1)]=s["db"+str(i+1)]/(1-np.power(beta2,t))
        
        parameters["W"+str(i+1)]=parameters["W"+str(i+1)]-(learning_rate*v_corrected["dW"+str(i+1)]/np.sqrt(s_corrected["dW"+str(i+1)]+epsilon))
        parameters["b"+str(i+1)]=parameters["b"+str(i+1)]-(learning_rate*v_corrected["db"+str(i+1)]/np.sqrt(s_corrected["db"+str(i+1)]+epsilon))
        
    return parameters, v, s







# Mini batch algorithm 
    
def random_mini_batches(X,Y,mini_batch_size,seed):
    
    np.random.seed(seed)
    m=X.shape[1] # number of examples
    mini_batches=[]
    
    # Setup random shuffle of training data
    permutation=list(np.random.permutation(m))
    shuffled_X=X[:,permutation]
    shuffled_Y=Y[:,permutation]
    
    # create mini batches from the shuffled 
    
    num_complete_minibatches=math.floor(m/mini_batch_size)
    
    for i in range(num_complete_minibatches):
        mini_batch_X=shuffled_X[:,i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch_Y=shuffled_Y[:,i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch=(mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
        
    if m % mini_batch_size !=0:
        mini_batch_X=shuffled_X[:,num_complete_minibatches*mini_batch_size:]
        mini_batch_Y=shuffled_Y[:,num_complete_minibatches*mini_batch_size:]
        mini_batch=(mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches








# Prediction algorithm 
    
def predict(X,Y,parameters):

    
    Y_prediction = np.zeros((Y.shape))
    AL, caches= L_model_forward(X,parameters)
    
    temp=np.amax(AL,axis=0).reshape((1,AL.shape[1]))
    Y_prediction[np.where(AL == temp)]=1
    Y_prediction[np.where(AL != temp)]=0
    
    accuracy= (100 - np.mean(np.abs(Y_prediction - Y)) * 100)
    
    return Y_prediction, accuracy








## Gradient Check algorithm
    
def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    key_list = list(parameters.keys())
    key_shape={}
    count = 0
    for key in key_list:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        #keys = keys + [key]*new_vector.shape[0]
        key_shape[key]=parameters[key].shape
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, key_shape,key_list

def vector_to_dictionary(theta,key_shape,key_list):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    a=0
    
    for key in key_list:
        b=key_shape[key][0]*key_shape[key][1]+a
        parameters[key] = theta[a:b].reshape(key_shape[key])
        a=b

    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    key_list=list(gradients.keys())
    count = 0
    
    for key in key_list:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta

def gradient_check(parameters,grad,X,Y):
    
    epsilon=1e-7
    # Set-up variables
    parameters_values,key_shape,key_list = dictionary_to_vector(parameters)
    #grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    for i in range(num_parameters):
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] = thetaplus[i][0]+epsilon
        AL, _ = L_model_forward(X, vector_to_dictionary(thetaplus,key_shape,key_list))
        J_plus[i]=compute_cost(AL,Y)
        
        thetaminus=np.copy(parameters_values)
        thetaminus[i][0]=thetaminus[i][0]-epsilon
        AL,_=L_model_forward(X,vector_to_dictionary(thetaminus,key_shape,key_list))
        J_minus[i]=compute_cost(AL,Y)
        
        
        gradapprox[i] = 0.5*(J_plus[i]-J_minus[i])/epsilon
    
    grad_temp=vector_to_dictionary(gradapprox,key_shape,key_list)
    for key in key_list:
        bp()
        numerator = np.linalg.norm(grad[key]-grad_temp[key])
        denominator = np.linalg.norm(grad[key])+np.linalg.norm(grad_temp[key])
        difference = numerator/denominator
    
        if difference > 2e-7:
            print ("\033[93m" + "There is a mistake in the backward propagation! " +"d" + str(key)+" difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! " +"d" + str(key)+" difference = " + str(difference) + "\033[0m")
    
        
        
    
    return difference
    
    
     
    
    
    