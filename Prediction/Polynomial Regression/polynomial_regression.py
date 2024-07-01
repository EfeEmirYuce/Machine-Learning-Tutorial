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
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg.predict(x),color="green")

plt.scatter(x,y,color="red")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)))

plt.scatter(x,y,color="red")
plt.plot(x,lin_reg3.predict(poly_reg3.fit_transform(x)),color="purple")


#predictions
print(lin_reg.predict([[6.6]]))
print(lin_reg.predict([[11]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))