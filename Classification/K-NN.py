#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#data import
datas = pd.read_csv("veriler.csv")

x = datas.iloc[:,1:4].values
y = datas.iloc[:,4:].values


#test traing splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=0)


#data scaling(standardization)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train_sc = sc.fit_transform(x_train)
x_test_sc = sc.transform(x_test)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(x_train_sc,y_train)

y_pred = logr.predict(x_test_sc)


#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")
knn.fit(x_train_sc,y_train)

y_pred = knn.predict(x_test_sc)

cm = confusion_matrix(y_test, y_pred)