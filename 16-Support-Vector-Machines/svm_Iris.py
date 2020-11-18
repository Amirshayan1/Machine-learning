# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 08:18:46 2020
SVM project based on Iris flower dataset
@author: Amirshayan Tatari
@email: sh.tatari18@gmail.com
"""

# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)

# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report


iris = sns.load_dataset('iris')

# Data exploration
sns.pairplot(iris, hue = 'species')
setosa = iris[iris['species']=='setosa']
plt.figure(figsize = (10, 8))
sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], 
            cmap = 'plasma', shade = True, shade_lowest = False)

# Creating datasets for training and testing
X = iris.drop('species', axis = 1)
Y = iris['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,
                                                    random_state = 101)

# Training and predictions
svc = SVC()
svc.fit(x_train, y_train)
predictions = svc.predict(x_test)

# Result report
conf_mat = confusion_matrix(y_test, predictions)
class_rep = classification_report(y_test, predictions)

# Tuning the model with grid search
param_grid = {'C':[0, 0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose = 3)
grid.fit(x_train, y_train)
grid.best_estimator_
grid_predict = grid.predict(x_test)

# Reslt: Grid search 
conf_mat_gs = confusion_matrix(y_test, grid_predict)