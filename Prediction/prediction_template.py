#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score


#data import
datas = pd.read_csv("maaslar.csv")


#data frame slicing
x = datas.iloc[:,1:2]
y = datas.iloc[:,2:]


#numPY array transform
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x,y)

print("Linear R2 value:")
print(r2_score(Y, lin_reg.predict(X)))


#polynomial regression(2nd degree)
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#polynomial regression(4th degree)
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#plotting(visualization)
'''
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg.predict(x),color="green")

plt.scatter(x,y,color="red")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)))

plt.scatter(x,y,color="red")
plt.plot(x,lin_reg3.predict(poly_reg.fit_transform(x)),color="purple")
'''

#predictions
print(lin_reg.predict([[6.6]]))
print(lin_reg.predict([[11]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

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

plt.scatter(x_scaled,y_scaled)
plt.plot(x_scaled,svr_reg.predict(x_scaled))
plt.show()

print(svr_reg.predict([[6.6]]))
print(svr_reg.predict([[11]]))

print("SVR R2 value:")
print(r2_score(y_scaled, svr_reg.predict(x_scaled)))


#decision tree
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

Z = X + 0.5
K = X - 0.4

plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")

plt.plot(x,r_dt.predict(Z),color="green")
plt.plot(x,r_dt.predict(K),color="yellow")
plt.show()

print(r_dt.predict([[6.6]]))
print(r_dt.predict([[11]]))

print("decision tree R2 value:")
print(r2_score(Y, r_dt.predict(X)))


#random forest
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X),color="blue")

plt.plot(X,rf_reg.predict(Z),color="green")
plt.plot(X,r_dt.predict(K),color="yellow")



#R^2
#from sklearn.metrics import r2_score
print("random forest R2 value:")
print(r2_score(Y, rf_reg.predict(X)))
print(r2_score(Y, rf_reg.predict(K)))
print(r2_score(Y, rf_reg.predict(Z)))