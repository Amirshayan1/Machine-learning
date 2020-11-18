# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:27:13 2020
A linear regression project with e-commerce data
@author: B.Sc. Amirshayan Tatari
@email: sh.tatari18@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Assigning dataframe
df = pd.read_csv('Ecommerce Customers')
df.head()
df.info()
 
# Data exploring
plt.figure(0)
sns.jointplot('Time on Website', 'Yearly Amount Spent', data = df)
plt.figure(1)
sns.jointplot('Time on App', 'Yearly Amount Spent', data = df)
plt.figure(2)
sns.jointplot('Time on App', 'Length of Membership', data = df, kind = "hex")
plt.figure(3)
sns.pairplot(df)

"""
It seems that the length of membership and yearly amount spent 
has a good correlation
"""
plt.figure(4)
sns.lmplot('Length of Membership', 'Yearly Amount Spent', data = df)

# Stablishing the training sets
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
Y = df['Yearly Amount Spent']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

# Training 
lm = LinearRegression()
lm.fit(x_train, y_train)
coef = lm.coef_
print(coef)

#Testing the model
predictions = lm.predict(x_test)

#Evaluation the predictions by the True/real data
plt.figure(5)
plt.scatter(y_test, predictions)
plt.xlabel('Y_test')
plt.ylabel('Predictions')

# Calculating the different form of errors
MAE = metrics.mean_absolute_error(y_test, predictions)
MSE = metrics.mean_squared_error(y_test, predictions)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, predictions))

# Residuals
plt.figure(6)
sns.distplot((y_test-predictions), bins = 50)
plt.xlabel('Years Amount Spent - Res.')

# Result
df_result = pd.DataFrame(coef, X.columns)
df_result.columns = ['Coeffecient']

