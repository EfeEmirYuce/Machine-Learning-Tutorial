#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

import statsmodels.api as sm

#data import
datas = pd.read_csv("maaslar_yeni.csv")


#data frame slicing
x = datas.iloc[:,2:5]
#x = datas.iloc[:,2:3]
y = datas.iloc[:,5:]

#numPY array transform
X = x.values
Y = y.values

print(datas.iloc[:,2:].corr())


#linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)

print("linear ols")
model = sm.OLS(lin_reg.predict(X),X)
print(model.fit().summary())

print("Linear R2 value:")
print(r2_score(Y, lin_reg.predict(X)))


#polynomial regression(4th degree)
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

print("polynomial ols")
model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

print("Polynomial R2 value:")
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))


#data scaling(standardization)
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_scaled = sc1.fit_transform(X)
sc2 = StandardScaler()
y_scaled = sc2.fit_transform(Y)


#support vector regression
from sklearn.svm import SVR

svr_reg = SVR(kernel = "rbf")
svr_reg.fit(x_scaled,y_scaled)


print("svr ols")
model3 = sm.OLS(svr_reg.predict(x_scaled),x_scaled)
print(model3.fit().summary())

print("SVR R2 value:")
print(r2_score(y_scaled, svr_reg.predict(x_scaled)))


#decision tree
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

Z = X + 0.5
K = X - 0.4


print("dt ols")
model4 = sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

print("decision tree R2 value:")
print(r2_score(Y, r_dt.predict(X)))


#random forest
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())


print("rf ols")
model5 = sm.OLS(rf_reg.predict(X),X)
print(model5.fit().summary())

#R^2
#from sklearn.metrics import r2_score
print("random forest R2 value:")
print(r2_score(Y, rf_reg.predict(X)))
print(r2_score(Y, rf_reg.predict(K)))
print(r2_score(Y, rf_reg.predict(Z)))