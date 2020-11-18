# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:30:48 2020
K Means Clustering method to cluster universities in two groups
@author: Amirshayan Tatari
@email: sh.tatari18@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('College_Data')
df.set_index('Unnamed: 0', drop = True, inplace = True)
df_des = df.describe()

# Data visualizations
sns.set_style('whitegrid')
plt.figure(figsize = (10,6))
sns.scatterplot(x = 'Room.Board', y = 'Grad.Rate', hue = 'Private', 
                data = df, alpha = 0.5)
plt.figure(figsize = (8,8))
sns.scatterplot(x = 'Outstate', y = 'F.Undergrad', hue = 'Private', 
                data = df, alpha = 0.5)
plt.figure(figsize = (11,7))
df[df['Private'] == 'Yes']['Outstate'].plot(kind = 'hist', 
                                            bins = 20, alpha = 0.6)
df[df['Private'] == 'No']['Outstate'].plot(kind = 'hist', 
                                           bins = 20, alpha = 0.8)
plt.xlabel('Outstate')
plt.figure(figsize = (11,7))
df[df['Private'] == 'Yes']['Grad.Rate'].plot(kind = 'hist',
                                             bins = 20, alpha = 0.6)
df[df['Private'] == 'No']['Grad.Rate'].plot(kind = 'hist',
                                            bins = 20, alpha = 0.8)
plt.xlabel('Grad.Rate')

# Checking data for unsual entries
err = df[df['Grad.Rate']>100]
df['Grad.Rate']['Cazenovia College'] = 100

plt.figure(figsize = (10, 8))
df[df['Private'] == 'Yes']['Grad.Rate'].plot(kind = 'hist', 
                                             bins = 20, alpha = 0.6)
df[df['Private'] == 'No']['Grad.Rate'].plot(kind = 'hist', 
                                            bins = 20, alpha = 0.7)
plt.xlabel('Grad.Rate')

# K Means Clustering unsupervised learning
kmeans = KMeans(n_clusters = 2)
kmeans.fit(df.drop('Private', axis = 1))
centers = kmeans.cluster_centers_

def converter (cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0 
df['cluster'] = df['Private'].apply(converter)

# Result evaluation
conf_mat = confusion_matrix(df['cluster'], kmeans.labels_)
class_rep = classification_report(df['cluster'], kmeans.labels_)