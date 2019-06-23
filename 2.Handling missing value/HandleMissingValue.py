#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:46:31 2019

@author: Arnold Jiadong Yu
@description: This script will handle missing values in two ways.
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training dataset
train = pd.read_csv('dataset/adult.data.txt', header = None)
# train.shape
# (32561,15) for train data
# train[1].value_counts()  # has ? 1843
# train[6].value_counts()  # has ? 1843
# train[13].value_counts() # has ? 583


# Importing the testing dataset
test = pd.read_csv('dataset/adult.test.txt', header = None)
# test.shape
# (16281,15) for test data
# test[1].value_counts()

train.columns = ['Age','Workclass','Fnlwgt','Education', 'Education-num','Marital','Occupation','Relationship','Race','Sex','CaptialG'\
              ,'CaptialL','Hours','Country','Income']

test.columns = ['Age','Workclass','Fnlwgt','Education', 'Education-num','Marital','Occupation','Relationship','Race','Sex','CaptialG'\
              ,'CaptialL','Hours','Country','Income']

# Remove Education-num attribute
train = train.drop(['Education-num' ], axis = 1)
test = test.drop(['Education-num' ], axis = 1)


# Taking care of the missing data

# 1: Remove all missing data
train = train.replace(' ?', np.nan).dropna(axis= 0)
# train.shape
#(30162,15) for train data
test = test.replace(' ?', np.nan).dropna(axis= 0)
# test.shape
#(15060, 15) for test data
test['Income'] = test['Income'].replace(' <=50K.', ' <=50K')
test['Income'] = test['Income'].replace(' >50K.', ' >50K')
# Export csv files for training and testing dataset
train.to_csv('train_removeMissing.csv', sep = ",", index = False)
test.to_csv('test_removeMissing.csv', sep = ",", index = False)

# 2: Replace all missing data with most frequence appeared data
train['Workclass'] = train['Workclass'].replace(' ?', ' Private')
# train['Workclass'].value_counts()
train['Occupation'] = train['Occupation'].replace(' ?', ' Prof-specialty')
# train['Occupation'].value_counts()
train['Country'] = train['Country'].replace(' ?', ' United-States')
# train['Country'].value_counts()

test['Workclass'] = test['Workclass'].replace(' ?', ' Private')
# test['Workclass'].value_counts()
test['Occupation'] = test['Occupation'].replace(' ?', ' Prof-specialty')
# test['Occupation'].value_counts()
test['Country'] = test['Country'].replace(' ?', ' United-States')
# test['Country'].value_counts(

test['Income'] = test['Income'].replace(' <=50K.', ' <=50K')
test['Income'] = test['Income'].replace(' >50K.', ' >50K')

train.to_csv('train_replaceMissing.csv', sep = ",", index = False)
test.to_csv('test_replaceMissing.csv', sep = ",", index = False)