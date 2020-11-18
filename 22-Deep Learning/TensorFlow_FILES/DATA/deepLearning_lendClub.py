# -*- coding: utf-8 -*-
"""
Keras API projcet using LendingClub dataset
@author: Amirshayan Tatari
@email: sh.tatari18@gmail.com
"""
"""
The goal is to predict whether a borrower will pay back the loan
 or not.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Dataframe
data_info = pd.read_csv('lending_club_info.csv',index_col='LoanStatNew')
df = pd.read_csv('lending_club_loan_two.csv')

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])
feat_info('mort_acc')

# Data visualizing
sns.set_style('whitegrid')
plt.figure(figsize = (7,7))
sns.countplot(x = 'loan_status', data = df)
plt.figure(figsize = (14,6))
df['loan_amnt'].plot(kind = 'hist', alpha = 0.7, bins = 40)

# Data correlation
corr = df.corr()
plt.figure(figsize = (10,10))
sns.heatmap(corr, cmap = 'Greens')

# feature: Installment exploration
feat_info('installment')
feat_info('loan_amnt')

plt.figure(figsize = (11,8))
sns.scatterplot(x = df['installment'], y = df['loan_amnt'])
plt.figure(figsize = (11,8))
sns.boxplot(x = df['loan_status'], y = df['loan_amnt'])

# Summary statistics for loan amount
summary1 = df.groupby('loan_status')['loan_amnt'].describe()

# Grade feature exploration
grade = sorted(df['grade'].unique())
sub_grade = sorted(df['sub_grade'].unique())

plt.figure(figsize = (11,7))
sns.countplot(x = 'grade', hue = 'loan_status', data = df)
plt.figure(figsize = (16,7))
sub_grade_order = sorted(df['sub_grade'].unique())
sns.countplot(x = 'sub_grade' , order = sub_grade_order, 
              data = df, hue = 'loan_status',alpha = 0.5)

# Providing variables for loan status
loan_repaid = pd.get_dummies(df['loan_status'], drop_first = True)
loan_repaid = loan_repaid.rename(columns = {'Fully Paid' : 'loan_repaid'})
df = pd.concat([df,loan_repaid], axis = 1)

# Data preprocessing
# Missing data
null_data = df.isnull().sum()/len(df)*100

feat_info('emp_title')
feat_info('emp_length')

job_title = df['emp_title'].value_counts()
job_title_num = len(df['emp_title'].unique())

# # Droping job title for having to many groups and causes confusion
# emp_length = sorted(df['emp_length'].dropna().unique())
# plt.figure(figsize = (11,8))
# sns.countplot(x = 'emp_length', data = df, hue = 'loan_status')

# Missing data drop
df.drop('emp_title', axis = 1, inplace = True)
df.drop('emp_length', axis = 1, inplace = True)
df.drop('title', axis = 1, inplace = True)
null_data = df.isnull().sum()

# Filling the missing data
feat_info('mort_acc')
num_mortAcc = df['mort_acc'].value_counts()
total_acc_mean = df.groupby('total_acc')['mort_acc'].mean()
def fill_mort (total_acc, mort_acc): 
    if np.isnan(mort_acc):
        return total_acc_mean[total_acc]
    else:
        return mort_acc
df['mort_acc'] = df.apply(lambda x: fill_mort(x['total_acc'], x['mort_acc']),
                          axis = 1)
df.dropna(inplace = True)
null_data = df.isnull().sum()

# Checking the data types
data_type = df.select_dtypes(['object']).columns

def term (x):
    item = x.split()[0]
    if item == '36':
        return 36
    else:
        return 60
df['term'] = df.apply(lambda x: term(x['term']), axis = 1)
df.drop('grade', axis = 1, inplace = True)

sub_grade = pd.get_dummies(df['sub_grade'], drop_first = True)
dummy_var = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose']],
                           drop_first = True)
df = pd.concat([df, sub_grade, dummy_var], axis = 1)
df.drop(['sub_grade', 'verification_status', 'application_type','initial_list_status','purpose'],
         axis = 1, inplace = True)
def home_own(x):
    if (x == 'ANY') | (x == 'NONE'):
        return 'OTHER'
    else:
        return x
df['home_ownership'] = df.apply(lambda x : home_own(x['home_ownership']), axis = 1)
home_owner = pd.get_dummies(df['home_ownership'], drop_first = True)
zip_code = df['address'].apply(lambda x: x.split()[-1])
code = pd.get_dummies(zip_code, drop_first = True)
cr_year = df['earliest_cr_line'].apply(lambda x: x.split('-')[1])
cr_year_dummy = pd.get_dummies(cr_year, drop_first = True)
df = pd.concat([df, code, home_owner, cr_year_dummy], axis = 1)
df.drop(['address', 'home_ownership', 'issue_d', 'earliest_cr_line', 'loan_status'], axis = 1,
        inplace = True)

# Providing training and test data
X = df.drop('loan_repaid', axis = 1).values
Y = df['loan_repaid'].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size = 0.2,
                                                    random_state = 101)

# Normalizing data
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Creating the NN model
model  = Sequential()

# Input layer
model.add(Dense(78, activation = 'relu'))
model.add(Dropout(0.2))

# Hidden layer
model.add(Dense(39, activation = 'relu'))
model.add(Dropout(0.2))

# Hidden layer
model.add(Dense(19, activation = 'relu'))
model.add(Dropout(0.2))

# Output layer
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam' )

# Training data
model.fit(x = x_train, y = y_train, epochs = 25,
          validation_data = (x_test, y_test))

# Model evaluation
losses = pd.DataFrame(model.history.history)
losses[['loss', 'val_loss']].plot()

predictions = model.predict_classes(x_test)

# Reports
conf_mat = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)

