"""

Decision tree vs random forest classifier 
based on LendingClub.com data in 2016
@author: Amirshayan Tatari
@email: sh.tatari18@gmail.com
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

loans = pd.read_csv('loan_data.csv')

# Data explorations
sns.set_style('whitegrid')
plt.figure(figsize = (10,6))
loans[loans['credit.policy'] == 1]['fico'].hist(alpha = 0.5, color = 'blue',
                                                bins = 30, label = 'Credit.Policy = 1')
loans[loans['credit.policy'] == 0]['fico'].hist(alpha = 0.5, color = 'red', 
                                                bins = 30, label = 'Credit.Policy = 0')
plt.legend()
plt.xlabel('FICO')

plt.figure(figsize = (10,6))
loans[loans['not.fully.paid'] == 1]['fico'].hist(alpha = 0.5, color = 'blue',
                                                 bins = 30, label = 'Not.fully.paid = 1')
loans[loans['not.fully.paid'] == 0]['fico'].hist(alpha = 0.5, color = 'red',
                                                 bins = 30, label = 'Not.fully.paid = 0')
plt.legend()
plt.xlabel('FICO')

plt.figure(figsize = (10,6))
sns.countplot(x = 'purpose', hue = 'not.fully.paid', data = loans)

plt.figure(figsize = (6,6))
sns.jointplot('fico', 'int.rate', data = loans)

plt.figure(figsize = (11,7))
sns.lmplot('fico', 'int.rate', data = loans, hue = 'credit.policy', col = 'not.fully.paid')

"""
Here I selected the not.fully.paid column for the classification model to predict
"""
"""
Decision Tree model
"""
# Creating final data for training  and testing
cat_feat = ['purpose']
final_data = pd.get_dummies(loans, columns = cat_feat, drop_first = True)
X = final_data.drop('not.fully.paid', axis = 1)
Y = final_data['not.fully.paid']
x_train, x_test, y_train, y_test = train_test_split(X,Y)

# Training and predictions
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
predictions_dtc = dtc.predict(x_test)

# Result evaluation
conf_mat_dtc = confusion_matrix(y_test, predictions_dtc)
print("Decision Tree:", "\n", classification_report(y_test, predictions_dtc))

"""
Random forest model
"""
# Training and predictions
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
predictions_rfc = rfc.predict(x_test)

# Result evaluation
conf_mat_rfc = confusion_matrix(y_test, predictions_rfc)
print("Random Forest:", "\n", classification_report(y_test, predictions_rfc))



