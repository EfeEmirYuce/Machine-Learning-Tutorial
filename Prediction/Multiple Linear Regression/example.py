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
datas = pd.read_csv("odev_tenis.csv")

#categorical(encoder : categoric -> numeric)
outlook = datas.iloc[:,0:1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
outlook[:,0] = le.fit_transform(datas.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

#categorical(encoder : categoric -> numeric)
windy = datas.iloc[:,3:4].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
windy[:,-1] = le.fit_transform(datas.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
windy = ohe.fit_transform(windy).toarray()

#categorical(encoder : categoric -> numeric)
play = datas.iloc[:,3:4].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
play[:,-1] = le.fit_transform(datas.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
play = ohe.fit_transform(play).toarray()

datas2 = datas.apply(preprocessing.LabelEncoder().fit_transform)


#data merging(dataframe concatenation)
r0 = pd.DataFrame(data=outlook, index=range(14), columns=["sunny","overcast","rainy"])

temperature = datas[["temperature"]]
r1 = pd.DataFrame(data=temperature, index=range(14), columns=["temperature"])

r2 = pd.DataFrame(data=windy[:,:1], index=range(14), columns=["windy"])

r3 = pd.DataFrame(data=play[:,:1], index=range(14), columns=["play"])

humidity = datas[["humidity"]]
r4 = pd.DataFrame(data=humidity, index=range(14), columns=["humidity"])

result1 = pd.concat([r0,r1], axis=1)
result2 = pd.concat([r2,r3], axis=1)
result3 = pd.concat([result2,r4], axis=1)

result_without_humidity = pd.concat([r0,result2], axis=1)

final_result = pd.concat([result1,result3], axis=1)

#test traing splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(result_without_humidity,humidity, test_size=0.33, random_state=0)


#humidity prediction
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


#backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((14,1)).astype(int), values = final_result, axis = 1)
X_l = final_result.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(humidity,X_l).fit()
print(model.summary())

final_result = final_result.iloc[:,1:]

X = np.append(arr = np.ones((14,1)).astype(int), values = final_result, axis = 1)
X_l = final_result.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(humidity,X_l).fit()
print(model.summary())

x_train = x_train.iloc[:1,:]
x_test = x_test.iloc[:1,:]

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)