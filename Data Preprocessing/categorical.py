#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:43:01 2024

@author: emir
"""

#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#data import
datas = pd.read_csv("eksikveriler.csv")


#data test
height = datas[["boy"]]

height_weight = datas[["boy","kilo"]]


#missing value
from sklearn.impute import SimpleImputer
#sci-kit learn
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") 
#replace missing values with the mean number
age = datas.iloc[:,1:4].values
imputer = imputer.fit(age[:,1:4])
age[:,1:4] = imputer.transform(age[:,1:4])


#categorical(encoder : categoric -> numeric)
country = datas.iloc[:,0:1].values
print(country)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(datas.iloc[:,0])
print(country)

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()
print(country)