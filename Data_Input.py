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

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def load_dataset():
    train_dataset = h5py.File('train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

layers_dims=[12288,25,12,6]

learning_rate = 0.0001
number_iter = 1500
minibatch_size = 32 
lambd = 2
keep_prob =1.2
print_cost = True
model = "L2_Reg"



parameters=L_layer_model_with_adam(X_train,Y_train,layers_dims,learning_rate,minibatch_size,number_iter,lambd,keep_prob,model,print_cost)


Y_pred_train, accuracy_train=predict(X_train,Y_train,parameters)
Y_pred_test, accuracy_test=predict(X_test,Y_test,parameters)
    
print("Training set accuracy = " + str(accuracy_train))
print("Test set accuracy = " + str(accuracy_test))



