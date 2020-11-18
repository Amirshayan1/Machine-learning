# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:15:21 2020
Logestic regression (Classification) project on advertising dataset
@author: B.Sc. Amirshayan Tatari
@email: sh.tatari18@gmail.com
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

ad_data = pd.read_csv('advertising.csv')
ad_data.describe()

# Data exploration
sns.set_style('whitegrid')
plt.figure(0)
sns.distplot(ad_data['Age'], bins = 30)
plt.figure(1)
sns.jointplot(ad_data['Age'], ad_data['Area Income'])
sns.jointplot(ad_data['Age'], ad_data['Area Income'], kind = 'kde')
sns.jointplot(ad_data['Daily Time Spent on Site'], ad_data['Daily Internet Usage'], color = 'green')
sns.pairplot(ad_data, hue ='Clicked on Ad')

'''
The dataframe includes columns that can be used as features to classify 
customers that click on the ad or not
'''

# Creating the train and test data
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
Y = ad_data['Clicked on Ad']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)

# Trainin the model
log_ml = LogisticRegression()
log_ml.fit(x_train, y_train)

# Predictions
predictions = log_ml.predict(x_test)

# Classificatio result
report  = classification_report(y_test, predictions)
conf_mat = confusion_matrix (y_test, predictions)
print(report, conf_mat)