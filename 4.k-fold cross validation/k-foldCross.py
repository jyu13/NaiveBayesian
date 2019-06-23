# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:06:07 2019

@author: Arnold Yu
@descripition: This script create class kFoldValidation.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from NaiveBayesionClassifier import NaiveBayesionClassifier as nb

class kFoldValidation:
    def __init__(self):
        pass
    
    # partition into 10 equal size data
    def kFold(self, df, k = 10):
        
        size = int(float(len(df))/float(k))
        # Partition into k of equal size 
        start = -1
        end = size
        dict_ = {}
        for a in range(k):
            dict_[a] = pd.DataFrame(df, index = list(range(start+1,end+1)))
            start += size
            end += size
        return dict_
    
    # calculate accuracy    
    def calculateAcc(self, matrix):
        denom = matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1]
        num = matrix[0][0] + matrix[1][1]
        return float(num)/float(denom)
    
    # calculate percision  
    def calculatePer(self, matrix):
        denom = matrix[0][0]+ matrix[1][0]
        return float(matrix[0][0])/float(denom)
    
    # calculate recall
    def calculateRec(self, matrix):
        denom = matrix[0][0]+ matrix[0][1]
        return float(matrix[0][0])/float(denom)
    
    # calculate F1
    def calculateF1(self, percision, recall):
        return float(2.0 * percision * recall)/ float(percision + recall)
    
    # calculate Matthew’s correlation coefficient
    def mcc(self, matrix):
        num = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
        denom = (( matrix[0][0] + matrix[0][1])*(matrix[0][0] + matrix[1][0])*( matrix[1][1] + matrix[0][1])*(matrix[1][1] + matrix[1][0]) ) ** 0.5
        return float(num)/float(denom)
       
    # validate each partition by running the classifier    
    def validationCate(self,df, dict_):
        
        list = []
        for i, j in dict_.items():
            test = j
            train = df[~df.index.isin(j.index)]
            classifier = nb()
            total_dict = classifier.trainingSetCategorical(train)
            matrix = classifier.testingSetCategorical(test, total_dict)
            accuary = self.calculateAcc(matrix)
            list.append(accuary)
        
        return list
    # validate each partition by running the classifier
    def validationCont(self,df, dict_):
        
        list = []
        for i, j in dict_.items():
            test = j
            train = df[~df.index.isin(j.index)]
            classifier = nb()
            total_dict = classifier.trainingSetGaussian(train)
            matrix = classifier.testingSetGaussian(test, total_dict)
            accuary = self.calculateAcc(matrix)
            list.append(accuary)
        
        return list
if __name__ == '__main__':
    train0 = pd.read_csv('dataset/train_removeMissing_cat.csv')
    test0 = pd.read_csv('dataset/test_removeMissing_cat.csv')
    result0 = pd.concat([train0,test0])
    kF0 = kFoldValidation()
    dict0 = kF0.kFold(train0)
    accuL0 = kF0.validationCate(train0, dict0)
    print("10-Fold on removeMissing_cat")
    print("Accuracy : " ,accuL0)
    print("Average Accuracy : ", float(sum(accuL0))/float(len(accuL0)))
    # 0.8088167
    
    train1 = pd.read_csv('dataset/train_replaceMissing_cat.csv')
    test1 = pd.read_csv('dataset/test_replaceMissing_cat.csv')
    result1 = pd.concat([train1,test1])
    kF1 = kFoldValidation()
    dict1 = kF1.kFold(train1)
    accuL1 = kF1.validationCate(train1, dict1)
    print("10-Fold on replaceMissing_cat")
    print("Accuracy : ", accuL1)
    print("Average Accuracy : ",float(sum(accuL1))/float(len(accuL1)))
    # 0.808842
    
    train2 = pd.read_csv('dataset/train_removeMissing.csv')
    test2 = pd.read_csv('dataset/test_removeMissing.csv')
    result2 = pd.concat([train2,test2])
    kF2 = kFoldValidation()
    dict2 = kF2.kFold(train2)
    accuL2 = kF2.validationCont(train2, dict2)
    print("10-Fold on removeMissing")
    print("Accuracy : ",accuL2)
    print("Average Accuracy : ",float(sum(accuL2))/float(len(accuL2)))
    #0.8109712
    
    train3 = pd.read_csv('dataset/train_replaceMissing.csv')
    test3 = pd.read_csv('dataset/test_replaceMissing.csv')
    result3 = pd.concat([train3,test3])
    kF3 = kFoldValidation()
    dict3 = kF3.kFold(train3)
    accuL3 = kF3.validationCont(train3, dict3)
    print("10-Fold on replaceMissing")
    print("Accuracy : ",accuL3)
    print("Average Accuracy : ",float(sum(accuL3))/float(len(accuL3)))
    #0.812748