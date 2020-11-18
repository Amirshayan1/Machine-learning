# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 07:57:57 2020
KNN classifier project
@author: B.Sc. Amirshayan Tatari
@email: sh.tatari18@gmail.com
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("KNN_Project_Data")
df.head()

# Data visualing
sns.pairplot(df, hue = 'TARGET CLASS')

# Standardizing the data
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis = 1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))
df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])

# Creating train data and test data
X = df_feat
Y = df['TARGET CLASS']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size =  0.3, random_state = 101)

# Checking different values for K
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    pred_i  = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.plot(range(1,40), error_rate, marker = 'o', markersize = 10)

# Learning 
knn = KNeighborsClassifier(n_neighbors = 27)
knn.fit(x_train, y_train)

# Predictions
predictions = knn.predict(x_test)

# Evaluation of the output
conf_mat = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)