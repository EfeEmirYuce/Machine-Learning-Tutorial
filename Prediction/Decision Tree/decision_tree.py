#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


#polynomial regression(2nd degree)
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


#polynomial regression(4th degree)
poly_reg3 = PolynomialFeatures(degree=4)
x_poly3 = poly_reg3.fit_transform(x)

lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)


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

print(r_dt.predict([[6.6]]))
print(r_dt.predict([[11]]))