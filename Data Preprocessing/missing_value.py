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
print(datas)


#data test
height = datas[["boy"]]

height_weight = datas[["boy","kilo"]]

#missing value
from sklearn.impute import SimpleImputer
#sci-kit learn
imputer = SimpleImputer(missing_values=np.nan, strategy="mean") 
#replace missing values with the mean number
yas = datas.iloc[:,1:4].values
print(yas)
imputer = imputer.fit(yas[:,1:4])
yas[:,1:4] = imputer.transform(yas[:,1:4])
print(yas)