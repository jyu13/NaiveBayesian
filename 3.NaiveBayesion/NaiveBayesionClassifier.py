# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 12:05:39 2019

@author: Arnold Yu
@description: This sript create class NaiveBayesionClassifier.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import math
import random

class NaiveBayesionClassifier:
    def __init__(self):
        pass
    # the pdf of normal distribution
    # f(y) = 1/(\sqrt(2pi)sigma) exp ( -(x-mu)^2/ 2 sigma^2)
    # where mean and varience comes from training set
    # x comes from test set
    def normalPDF(self, x, mean, var):
        denom = (2 * math.pi * var) ** .05
        num = math.exp(-(float(x)-float(mean)) ** 2 / (2 * var))
        return num/denom

    def calculateMeanVar(self, df):
        sum_ = sum(df)
        length = len(df)
        mean = float(sum_)/float(length)
        sum_V = 0.0
        for idx, j in np.ndenumerate(df):
            sum_V += (j - mean) ** 2.0
    
        # here we calculate unbiased varience S^2 instead of sigma^2
        varience = sum_V / float(length-1)
    
        # return tuple of mean and varience
        return (mean, varience)

    def trainingSetCategorical(self, df):
        # Spliting binary dependent variables
        data_Greater50k = df[df['Income'] == ' >50K']
        data_LessEqual50k = df[df['Income'] == ' <=50K']
        
        # Get length of data_Greater50k and data_LessEqual50k
        len_Greater50k = len(data_Greater50k)
        len_LessEqual50k = len(data_LessEqual50k)
        
        # Get probability of dependent variables
        # Probability of income greater than 50k
        prob_Greater50k = float(len_Greater50k)/float(len_Greater50k + len_LessEqual50k)  
        # probability of income less and equal than 50k
        prob_LessEqual50k = 1.0 - prob_Greater50k
        
        # The goal is to find P(X|income = ' >50k') and P(X|income = ' <=50k') 
        # where X reperesents independent variables
        
        # Doing laplacian
        age_labels1 = {'A1': 1,'A2': 1,'A3': 1,'A4': 1,'A5': 1,'A6': 1,'A7': 1,'A8': 1,'A9': 1,'A10': 1}
        fnlwgt_labels1 = {'F0': 1,'F1': 1,'F2': 1,'F3': 1,'F4': 1,'F5': 1,'F6': 1,'F7': 1,'F8': 1,'F9': 1,'F10': 1,'F11': 1,'F12': 1,'F13': 1,'F14': 1}
        captialG_labels1 = {'CG0': 1,'CG1': 1,'CG2': 1,'CG3': 1,'CG4': 1,'CG5': 1,'CG6': 1,'CG7': 1,'CG8': 1,'CG9': 1,'CG10': 1,'CG11': 1}
        captialL_labels1 = {'CL0': 1,'CL1': 1,'CL2': 1,'CL3': 1,'CL4': 1,'CL5': 1,'CL6': 1,'CL7': 1,'CL8': 1,'CL9': 1}
        hours_labels1 = {'H0': 1,'H1': 1,'H2': 1,'H3': 1,'H4': 1,'H5': 1,'H6': 1,'H7': 1,'H8': 1,'H9': 1}
        age_labels2 = {'A1': 1,'A2': 1,'A3': 1,'A4': 1,'A5': 1,'A6': 1,'A7': 1,'A8': 1,'A9': 1,'A10': 1}
        fnlwgt_labels2 = {'F0': 1,'F1': 1,'F2': 1,'F3': 1,'F4': 1,'F5': 1,'F6': 1,'F7': 1,'F8': 1,'F9': 1,'F10': 1,'F11': 1,'F12': 1,'F13': 1,'F14': 1}
        captialG_labels2 = {'CG0': 1,'CG1': 1,'CG2': 1,'CG3': 1,'CG4': 1,'CG5': 1,'CG6': 1,'CG7': 1,'CG8': 1,'CG9': 1,'CG10': 1,'CG11': 1}
        captialL_labels2 = {'CL0': 1,'CL1': 1,'CL2': 1,'CL3': 1,'CL4': 1,'CL5': 1,'CL6': 1,'CL7': 1,'CL8': 1,'CL9': 1}
        hours_labels2 = {'H0': 1,'H1': 1,'H2': 1,'H3': 1,'H4': 1,'H5': 1,'H6': 1,'H7': 1,'H8': 1,'H9': 1}
        set_ = ['Income', 'Age_cat', 'Fnlwgt_cat','CaptialG_cat', 'CaptialL_cat','Hours_cat']
        
        dict_1 = {}
        dict_2 = {}
        # Use dictionary to calculate probability of each entry in each attribute.
        # The following assume all attributes are categorical data.
        for column in data_Greater50k:
            if column not in set_:
        
                temp1 = Counter(data_Greater50k[column])
                # Calculate all probabilities in Greater50k category 
                # Ex. P( Age = 35 | income = >50k) = #(Age = 35, income  = >50k) / # (income = >50k)
                for i, j in temp1.items():
                    temp1[i] = float(j)/ float(len_Greater50k)
                dict_1[column] = temp1
              
                
                temp2 = Counter(data_LessEqual50k[column])    
                # Calculate all probabilities in LessEqual50k category 
                for i, j in temp2.items():
                    temp2[i] = float(j)/ float(len_LessEqual50k)
                dict_2[column] = temp2
                
                
            if column == 'Age_cat':
                temp1 = Counter(data_Greater50k[column])          
                for i, j in temp1.items():
                    age_labels1[i] = age_labels1[i] + j
                for i, j in age_labels1.items():
                    age_labels1[i] = float(j)/ float(len_Greater50k + 10)
                dict_1[column] = age_labels1
                temp2 = Counter(data_LessEqual50k[column])
            
                for i, j in temp2.items():
                    age_labels2[i] = age_labels2[i] + j
                for i, j in age_labels2.items():
                    age_labels2[i] = float(j)/ float(len_LessEqual50k + 10)
                dict_2[column] = age_labels2
                
                
                
            if column == 'Fnlwgt_cat':
                temp1 = Counter(data_Greater50k[column])          
                for i, j in temp1.items():
                    fnlwgt_labels1[i] = fnlwgt_labels1[i] + j
                for i, j in fnlwgt_labels1.items():
                    fnlwgt_labels1[i] = float(j)/ float(len_Greater50k + 15)
                dict_1[column] = fnlwgt_labels1
                temp2 = Counter(data_LessEqual50k[column])
            
                for i, j in temp2.items():
                    fnlwgt_labels2[i] = fnlwgt_labels2[i] + j
                for i, j in fnlwgt_labels2.items():
                    fnlwgt_labels2[i] = float(j)/ float(len_LessEqual50k + 15)
                dict_2[column] = fnlwgt_labels2
                
                
                
            if column == 'CaptialG_cat':
                temp1 = Counter(data_Greater50k[column])          
                for i, j in temp1.items():
                    captialG_labels1[i] = captialG_labels1[i] + j
                for i, j in captialG_labels1.items():
                    captialG_labels1[i] = float(j)/ float(len_Greater50k + 12)
                dict_1[column] = captialG_labels1
                temp2 = Counter(data_LessEqual50k[column])
            
                for i, j in temp2.items():
                    captialG_labels2[i] = captialG_labels2[i] + j
                for i, j in captialG_labels2.items():
                    captialG_labels2[i] = float(j)/ float(len_LessEqual50k + 12)
                dict_2[column] = captialG_labels2
                
                
                
            if column == 'CaptialL_cat':
                temp1 = Counter(data_Greater50k[column])          
                for i, j in temp1.items():
                    captialL_labels1[i] = captialL_labels1[i] + j
                for i, j in captialL_labels1.items():
                    captialL_labels1[i] = float(j)/ float(len_Greater50k + 10)
                dict_1[column] = captialL_labels1
                temp2 = Counter(data_LessEqual50k[column])
            
                for i, j in temp2.items():
                    captialL_labels2[i] = captialL_labels2[i] + j
                for i, j in captialL_labels2.items():
                    captialL_labels2[i] = float(j)/ float(len_LessEqual50k + 10)
                dict_2[column] = captialL_labels2
                
                
            if column == 'Hours_cat':
                temp1 = Counter(data_Greater50k[column])          
                for i, j in temp1.items():
                    hours_labels1[i] = hours_labels1[i] + j
                for i, j in hours_labels1.items():
                    hours_labels1[i] = float(j)/ float(len_Greater50k + 10)
                dict_1[column] = hours_labels1
                temp2 = Counter(data_LessEqual50k[column])
            
                for i, j in temp2.items():
                    hours_labels2[i] = hours_labels2[i] + j
                for i, j in hours_labels2.items():
                    hours_labels2[i] = float(j)/ float(len_LessEqual50k + 10)
                dict_2[column] = hours_labels2
                
                
        dict_1[' >50K'] = prob_Greater50k
        dict_2[' <=50K'] = prob_LessEqual50k
            
        total_dict = {' >50K' : dict_1 , ' <=50K' : dict_2}
        
        return total_dict
                


    def testingSetCategorical(self, df, total_dict):
        # set up confusion matrix 
        #         predicted <=50k  |   predicted > 50k
        # <=50k |
        # >50k  |
        
        confusionMatrix = [[0,0],[0,0]]
        predicted = []
        i = 0
        # P(income | X ) = P(X | income) * P(income) / P(X)
        # P(income | X ) is propotion of P(X| income) * P(income)
        
        
    
        for row in df.iterrows():
            prob_Greater50k = 1.0
            prob_LessEqual50k = 1.0
            for j in df:
                if j != 'Income':
                    # P(X | income = >50 )
                    value = row[1][j]
                    if value in total_dict[' >50K'][j]:           
                        prob_Greater50k = prob_Greater50k * total_dict[' >50K'][j][value]
    
                    # P( X | income = <= 50 )
                    if value in total_dict[' <=50K'][j]:
                        prob_LessEqual50k = prob_LessEqual50k * total_dict[' <=50K'][j][value]

    
                # P(X| income) * P(income)
            prob_Greater50k = prob_Greater50k * total_dict[' >50K'][' >50K']
            prob_LessEqual50k = prob_LessEqual50k * total_dict[' <=50K'][' <=50K']
    
            if prob_Greater50k > prob_LessEqual50k:
                predicted.append(' >50K')
            else:
                predicted.append(' <=50K')
        
            if row[1]['Income'] == ' <=50K' and predicted[i] == ' <=50K':
                confusionMatrix[0][0] += 1
                i += 1
            elif row[1]['Income'] == ' <=50K' and predicted[i] == ' >50K':
                confusionMatrix[0][1] += 1
                i += 1
            elif row[1]['Income'] == ' >50K' and predicted[i] == ' <=50K':
                confusionMatrix[1][0] += 1
                i += 1
            else:
                confusionMatrix[1][1] += 1
                i += 1
        return confusionMatrix   
                
                
                
                
    def trainingSetGaussian(self, df):
        # Spliting binary dependent variables
        data_Greater50k = df[df['Income'] == ' >50K']
        data_LessEqual50k = df[df['Income'] == ' <=50K']
            
        # Get length of data_Greater50k and data_LessEqual50k
        len_Greater50k = len(data_Greater50k)
        len_LessEqual50k = len(data_LessEqual50k)
            
        # Get probability of dependent variables
        # Probability of income greater than 50k
        prob_Greater50k = float(len_Greater50k)/float(len_Greater50k + len_LessEqual50k)  
        # probability of income less and equal than 50k
        prob_LessEqual50k = 1.0 - prob_Greater50k
            
        # The goal is to find P(X|income = ' >50k') and P(X|income = ' <=50k') 
        # where X reperesents independent variables
            
        set_ = ['Age', 'Fnlwgt', 'CaptialG', 'CaptialL', 'Hours']
        dict_1 = {}
        dict_2 = {}
            
        # Use dictionary to calculate probability of each entry in each attribute.
        # The following assume all attributes are categorical data.
        for column in data_Greater50k:
            if column != 'Income' and column not in set_:
            
                temp1 = Counter(data_Greater50k[column])
                # Calculate all probabilities in Greater50k category 
                # Ex. P( Age = 35 | income = >50k) = #(Age = 35, income  = >50k) / # (income = >50k)
                for i, j in temp1.items():
                    temp1[i] = float(j)/ float(len_Greater50k)
                dict_1[column] = temp1
                  
                    
                temp2 = Counter(data_LessEqual50k[column])    
                # Calculate all probabilities in LessEqual50k category 
                for i, j in temp2.items():
                    temp2[i] = float(j)/ float(len_LessEqual50k)
                dict_2[column] = temp2
                # Check if the attribute is continuous, then use Gaussion distribution
            if column in set_:
                meanAndV1 = self.calculateMeanVar(data_Greater50k[column])
                meanAndV2 = self.calculateMeanVar(data_LessEqual50k[column])
                dict_1[column] = meanAndV1
                dict_2[column] = meanAndV2
                    
        dict_1[' >50K'] = prob_Greater50k
        dict_2[' <=50K'] = prob_LessEqual50k
            
        total_dict = {' >50K' : dict_1 , ' <=50K' : dict_2}
        
        return total_dict  
        
    def testingSetGaussian(self, df, total_dict):
        confusionMatrix = [[0,0],[0,0]]
        predicted = []
        i = 0
        # P(income | X ) = P(X | income) * P(income) / P(X)
        # P(income | X ) is propotion of P(X| income) * P(income)
        
        set_ = ['Age', 'Fnlwgt', 'CaptialG', 'CaptialL', 'Hours']
    
        for row in df.iterrows():
            prob_Greater50k = 1.0
            prob_LessEqual50k = 1.0
            for j in df:
                if j != 'Income' and j not in set_:
                    # P(X | income = >50 )
                    value = row[1][j]
                    if value in total_dict[' >50K'][j]:           
                        prob_Greater50k = prob_Greater50k * total_dict[' >50K'][j][value]
    
                    # P( X | income = <= 50 )
                    if value in total_dict[' <=50K'][j]:
                        prob_LessEqual50k = prob_LessEqual50k * total_dict[' <=50K'][j][value]

                if j in set_:
                    value = row[1][j]
                    prob_Greater50k = prob_Greater50k * self.normalPDF(value,total_dict[' >50K'][j][0],total_dict[' >50K'][j][1] )
                    prob_LessEqual50k = prob_LessEqual50k * self.normalPDF(value,total_dict[' <=50K'][j][0],total_dict[' <=50K'][j][1] )
                # P(X| income) * P(income)
            prob_Greater50k = prob_Greater50k * total_dict[' >50K'][' >50K']
            prob_LessEqual50k = prob_LessEqual50k * total_dict[' <=50K'][' <=50K']
    
            if prob_Greater50k > prob_LessEqual50k:
                predicted.append(' >50K')
            else:
                predicted.append(' <=50K')
        
            if row[1]['Income'] == ' <=50K' and predicted[i] == ' <=50K':
                confusionMatrix[0][0] += 1
                i += 1
            elif row[1]['Income'] == ' <=50K' and predicted[i] == ' >50K':
                confusionMatrix[0][1] += 1
                i += 1
            elif row[1]['Income'] == ' >50K' and predicted[i] == ' <=50K':
                confusionMatrix[1][0] += 1
                i += 1
            else:
                confusionMatrix[1][1] += 1
                i += 1
        return confusionMatrix  
    
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
if __name__ == '__main__':
    
    # Modeling all categorical attribute
    train = pd.read_csv('dataset/train_removeMissing_cat.csv')
    test = pd.read_csv('dataset/test_removeMissing_cat.csv')    
    classifier = NaiveBayesionClassifier()
    data = classifier.trainingSetCategorical(train)
    matrix = classifier.testingSetCategorical(test,data)
    print("RemoveMissing_cat")
    print("Matrix : ", matrix)
    accu = classifier.calculateAcc(matrix)
    print('%s%f' %("Accuracy : ", accu))
    percision = classifier.calculatePer(matrix)
    print('%s%f' %("Percision : ", percision))
    recall = classifier.calculateRec(matrix)
    print('%s%f' %("Recall : ", recall))
    f1 = classifier.calculateF1(percision, recall)
    print('%s%f' %("F1 : ", f1))
    mcc = classifier.mcc(matrix)
    print('%s%f' %("Matthew’s correlation coefficient : ", mcc))
    # accuracy 0.8098 
    # accuracy 0.8076 without Workclass
    # accuracy 0.8091 without Country
    
    
    train1 = pd.read_csv('dataset/train_replaceMissing_cat.csv')
    test1 = pd.read_csv('dataset/test_replaceMissing_cat.csv')
    classifier1 = NaiveBayesionClassifier()
    data1 = classifier1.trainingSetCategorical(train1)
    matrix1 = classifier1.testingSetCategorical(test1,data1)
    print("ReplaceMissing_cat")
    print("Matrix : ", matrix1)
    accu1 = classifier.calculateAcc(matrix1)
    print('%s%f' %("Accuracy : ", accu1))
    percision1 = classifier.calculatePer(matrix1)
    print('%s%f' %("Percision : ", percision1))
    recall1 = classifier.calculateRec(matrix1)
    print('%s%f' %("Recall : ", recall1))
    f11 = classifier.calculateF1(percision1, recall1)
    print('%s%f' %("F1 : ", f11))
    mcc1 = classifier.mcc(matrix1)
    print('%s%f' %("Matthew’s correlation coefficient : ", mcc1))
    # accuracy 0.8119
    
    # Modeling categorical and continuous attributes
    train2 = pd.read_csv('dataset/train_removeMissing.csv')
    test2 = pd.read_csv('dataset/test_removeMissing.csv')
    classifier2 = NaiveBayesionClassifier()
    data2 = classifier2.trainingSetGaussian(train2)  
    matrix2 = classifier2.testingSetGaussian(test2,data2)
    print("RemoveMissing")
    print("Matrix : ", matrix2)
    accu2 = classifier.calculateAcc(matrix2)
    print('%s%f' %("Accuracy : ", accu2))
    percision2 = classifier.calculatePer(matrix2)
    print('%s%f' %("Percision : ", percision2))
    recall2 = classifier.calculateRec(matrix2)
    print('%s%f' %("Recall : ", recall2))
    f12 = classifier.calculateF1(percision2, recall2)
    print('%s%f' %("F1 : ", f12))
    mcc2 = classifier.mcc(matrix2)
    print('%s%f' %("Matthew’s correlation coefficient : ", mcc2))
    # accuracy 0.8079
    
    
    train3 = pd.read_csv('dataset/train_replaceMissing.csv')
    test3 = pd.read_csv('dataset/test_replaceMissing.csv')
    classifier3 = NaiveBayesionClassifier()
    data3 = classifier2.trainingSetGaussian(train3)  
    matrix3 = classifier2.testingSetGaussian(test3,data3)
    print("ReplaceMissing")
    print("Matrix : ", matrix3)
    accu3 = classifier.calculateAcc(matrix3)
    print('%s%f' %("Accuracy : ", accu3))
    percision3 = classifier.calculatePer(matrix3)
    print('%s%f' %("Percision : ", percision3))
    recall3 = classifier.calculateRec(matrix3)
    print('%s%f' %("Recall : ", recall3))
    f13 = classifier.calculateF1(percision3, recall3)
    print('%s%f' %("F1 : ", f13))
    mcc3 = classifier.mcc(matrix3)
    print('%s%f' %("Matthew’s correlation coefficient : ", mcc3))
    # accuracy 0.8112