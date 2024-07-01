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
datas = pd.read_csv("veriler.csv")


#categorical(encoder : categoric -> numeric)
country = datas.iloc[:,0:1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
country[:,0] = le.fit_transform(datas.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
country = ohe.fit_transform(country).toarray()

#categorical(encoder : categoric -> numeric)
gender = datas.iloc[:,-1:].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
gender[:,-1] = le.fit_transform(datas.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
gender = ohe.fit_transform(gender).toarray()


#data merging(dataframe concatenation)
result = pd.DataFrame(data=country, index=range(22), columns=["fr","tr","us"])

height_weight_age = datas[["boy","kilo","yas"]]
result2 = pd.DataFrame(data=height_weight_age, index=range(22), columns=["boy","kilo","yas"])

result3 = pd.DataFrame(data=gender[:,:1], index=range(22), columns=["cinsiyet"])

r = pd.concat([result,result2], axis=1)
r2 = pd.concat([r,result3], axis=1)


#test traing splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(r,result3, test_size=0.33, random_state=0)


#data scaling(standardization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#gender prediction
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


#height prediction
height = r2.iloc[:,3:4].values
left = r2.iloc[:,:3]
right = r2.iloc[:,4:]

data = pd.concat([left,right],axis=1)

x_train, x_test, y_train, y_test = train_test_split(data,height, test_size=0.33, random_state=0)

regressor2 = LinearRegression()
regressor2.fit(x_train, y_train)

y_pred2 = regressor2.predict(x_test)


#backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((22,1)).astype(int), values = data, axis = 1)
X_l = data.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(height,X_l).fit()
print(model.summary())

X_l = data.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(height,X_l).fit()
print(model.summary())

X_l = data.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(height,X_l).fit()
print(model.summary())