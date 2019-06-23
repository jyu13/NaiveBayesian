# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:26:13 2019

@author: Arnold Yu
@description: This python script will convert continuous attributes to categorical attributes
"""



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set

train = pd.read_csv('train_removeMissing.csv')
train1 = pd.read_csv('train_replaceMissing.csv')
# Importing the testing set
test = pd.read_csv('test_removeMissing.csv')
test1 = pd.read_csv('test_replaceMissing.csv')


age_bins = [0,10,20,30,40,50,60,70,80,90,100]
age_labels = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10']

train['Age_cat'] = pd.cut(train['Age'], age_bins, labels=age_labels, right=True, include_lowest=True)
train = train.drop(['Age'], axis = 1)
train1['Age_cat'] = pd.cut(train1['Age'], age_bins, labels=age_labels, right=True, include_lowest=True)
train1 = train1.drop(['Age'], axis = 1)

test['Age_cat'] = pd.cut(test['Age'], age_bins, labels=age_labels, right=True, include_lowest=True)
test = test.drop(['Age'], axis = 1)
test1['Age_cat'] = pd.cut(test1['Age'], age_bins, labels=age_labels, right=True, include_lowest=True)
test1 = test1.drop(['Age'], axis = 1)


fnlwgt_bins = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
fnlwgt_labels = ['F0','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14']

train['Fnlwgt_cat'] = pd.cut(train['Fnlwgt'], fnlwgt_bins, labels = fnlwgt_labels, right=True, include_lowest=True)
train = train.drop(['Fnlwgt'], axis = 1)
train1['Fnlwgt_cat'] = pd.cut(train1['Fnlwgt'], fnlwgt_bins, labels = fnlwgt_labels, right=True, include_lowest=True)
train1 = train1.drop(['Fnlwgt'], axis = 1)

test['Fnlwgt_cat'] = pd.cut(test['Fnlwgt'], fnlwgt_bins, labels = fnlwgt_labels, right=True, include_lowest=True)
test = test.drop(['Fnlwgt'], axis = 1)
test1['Fnlwgt_cat'] = pd.cut(test1['Fnlwgt'], fnlwgt_bins, labels = fnlwgt_labels, right=True, include_lowest=True)
test1 = test1.drop(['Fnlwgt'], axis = 1)

captialG_bins = [0, 9000, 18000, 27000, 36000, 45000, 54000, 63000, 72000, 81000, 90000, 99000, 150000]
captialG_labels = ['CG0','CG1','CG2','CG3','CG4','CG5','CG6','CG7','CG8','CG9','CG10','CG11']

train['CaptialG_cat'] = pd.cut(train['CaptialG'], captialG_bins, labels = captialG_labels, right=True, include_lowest=True)
train = train.drop(['CaptialG'], axis = 1)
train1['CaptialG_cat'] = pd.cut(train1['CaptialG'], captialG_bins, labels = captialG_labels, right=True, include_lowest=True)
train1 = train1.drop(['CaptialG'], axis = 1)

test['CaptialG_cat'] = pd.cut(test['CaptialG'], captialG_bins, labels = captialG_labels, right=True, include_lowest=True)
test =test.drop(['CaptialG'], axis = 1)
test1['CaptialG_cat'] = pd.cut(test1['CaptialG'], captialG_bins, labels = captialG_labels, right=True, include_lowest=True)
test1 = test1.drop(['CaptialG'], axis = 1)

captialL_bin = [0, 450, 900, 1350, 1800, 2250, 2700, 3150, 3600, 4050, 9000]
captialL_labels = ['CL0','CL1','CL2','CL3','CL4','CL5','CL6','CL7','CL8','CL9']

train['CaptialL_cat'] = pd.cut(train['CaptialL'], captialL_bin, labels = captialL_labels, right=True, include_lowest=True)
train = train.drop(['CaptialL'], axis = 1)
train1['CaptialL_cat'] = pd.cut(train1['CaptialL'], captialL_bin, labels = captialL_labels, right=True, include_lowest=True)
train1 = train1.drop(['CaptialL'], axis = 1)

test['CaptialL_cat'] = pd.cut(test['CaptialL'], captialL_bin, labels = captialL_labels, right=True, include_lowest=True)
test = test.drop(['CaptialL'], axis = 1)
test1['CaptialL_cat'] = pd.cut(test1['CaptialL'], captialL_bin, labels = captialL_labels, right=True, include_lowest=True)
test1 = test1.drop(['CaptialL'], axis = 1)

hours_bin = [0, 10,20,30,40,50,60,70,80,90,150]
hours_labels = ['H0','H1','H2','H3','H4','H5','H6','H7','H8','H9']

train['Hours_cat'] = pd.cut(train['Hours'], hours_bin, labels = hours_labels, right=True, include_lowest=True)
train = train.drop(['Hours'], axis = 1)
train1['Hours_cat'] = pd.cut(train1['Hours'], hours_bin, labels = hours_labels, right=True, include_lowest=True)
train1 = train1.drop(['Hours'], axis = 1)

test['Hours_cat'] = pd.cut(test['Hours'], hours_bin, labels = hours_labels, right=True, include_lowest=True)
test = test.drop(['Hours'], axis = 1)
test1['Hours_cat'] = pd.cut(test1['Hours'], hours_bin, labels = hours_labels, right=True, include_lowest=True)
test1 = test1.drop(['Hours'], axis = 1)



train.to_csv('train_removeMissing_cat.csv', sep = ",", index = False)
test.to_csv('test_removeMissing_cat.csv', sep = ",", index = False)

train1.to_csv('train_replaceMissing_cat.csv', sep = ",", index = False)
test1.to_csv('test_replaceMissing_cat.csv', sep = ",", index = False) 