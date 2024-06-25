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
datas = pd.read_csv("satislar.csv")
print(datas)

months = datas[["Aylar"]]
sales = datas[["Satislar"]]
print(months)
print(sales)

sales2 = datas.iloc[:,1:2].values
print(sales2)


#test traing splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(months,sales, test_size=0.33, random_state=0)


#data scaling(standardization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)