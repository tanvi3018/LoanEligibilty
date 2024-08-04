#!/usr/bin/env python
# coding: utf-8

# # Classification Agorithms

# ## **Loan Eligibility Prediction Model**

# ### Project Scope:
# 
# Loans form an integral part of banking operations. However, not all loans are returned and hence it is important for a bank to closely moniter its loan applications. This case study is an analysis of the German Credit data. It contains details of 614 loan applicants with 13 attributes and the classification whether an applicant was granted loan or denied loan.
# 
# **Your role:** Using the available dataset, train a classification model to predict whether an applicant should be given loan.
# 
# **Goal:** Build a model to predict loan eligiblity with an average acuracy of more than 76%
# 
# **Specifics:** 
# 
# * Machine Learning task: Classification model 
# * Target variable: Loan_Status 
# * Input variables: Refer to data dictionary below
# * Success Criteria: Accuracy of 76% and above
# 

# ## Data Dictionary:
# 
# * **Loan_ID:** Applicant ID
# * **Gender:** Gender of the applicant Male/Female
# * **Married:** Marital status of the applicant
# * **Dependents:** Number of dependants the applicant has
# * **Education:** Highest level of education
# * **Self_Employed:** Whether self-employed Yes/No
# * **ApplicantIncome:** Income of the applicant
# * **CoapplicantIncome:** Income of the co-applicant
# * **LoanAmount:** Loan amount requested
# * **Loan_Amount_Term:** Term of the loan
# * **Credit_History:** Whether applicant has a credit history
# * **Property_Area:** Current property location
# * **Loan_Approved:** Loan approved yes/no

# ## **Data Analysis and Data Prep**

# ### Loading all the necessary packages

# In[60]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# ### Reading the data

# In[61]:


# Import the data from 'credit.csv'
df = pd.read_csv('credit.csv')
df.head(5)


# In[62]:


# check the number of rows and observations
df.shape


# In[63]:


# How many application were approved and how many were denied?
df['Loan_Approved'].value_counts().plot.bar()


# 422 people (around 69%) out of 614 were eligible for loan

# ### Missing value imputation

# In[64]:


# check for missing values in each variable
df.isnull().sum()


# Consider these methods to fill in the missing values:
# * For numerical variables: imputate using mean or median 
# * For categorical variables: imputate using mode
# 
# For e.g.
# In the `Loan_Amount_Term` variable, the value of 360 is repeating the most. 
# 
# You can check that by using `train['Loan_Amount_Term'].value_counts()`
# 
# So you will replace the missing values in this variable using the mode of this variable. i.e. 360
# 
# 
# 
# For the `LoanAmount` variable, check if the variable has ouliers by plotting a box plot. If there are outliers use the median to fill the null values since mean is highly affected by the presence of outliers. If there are no outliers use mean to impute missing values in `LoanAmount'

#     

# In[65]:


df.dtypes


# In[66]:


df['Dependents'].mode()[0]


# In[67]:


sns.distplot(df['LoanAmount'])


# In[68]:


# impute all missing values in all the features

df['Gender'].fillna('Male', inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)


# In[69]:


# Confirm if there are any missing values left
df.isnull().sum()


# ### Data Prep

# In[70]:


# drop 'Loan_ID' variable from the data. We won't need it.
df = df.drop('Loan_ID', axis=1)


# In[71]:


df.dtypes


# In[72]:


# Create dummy variables for all 'object' type variables except 'Loan_Status'
df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents','Education','Self_Employed','Property_Area'])
df.head(2)


# In[73]:


# saving this procewssed dataset
df.to_csv('Processed_Credit_Dataset.csv', index=None)


# ### Data Partition

# In[27]:


# Seperate the input features and target variable
x = df.drop('Loan_Approved',axis=1)
y = df.Loan_Approved


#     

# In[28]:


# splitting the data in training and testing set
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2, random_state=123)


# In[29]:


xtrain.shape, xtest.shape, ytrain.shape, ytest.shape


# In[30]:


# scale the data using min-max scalar
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()


# In[31]:


# Fit-transform on train data
xtrain_scaled = scale.fit_transform(xtrain)
xtest_scaled = scale.transform(xtest)


#     

#     

# # **Models**

# ## <font color='chocolate'>**1. Logistic Regression**</font>

# In[32]:


from sklearn.linear_model import LogisticRegression

lrmodel = LogisticRegression().fit(xtrain_scaled, ytrain)


# In[35]:


# Predict the loan eligibility on testing set and calculate its accuracy.
# First, from sklearn.metrics import accuracy_score and confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix

ypred = lrmodel.predict(xtest_scaled)

accuracy_score(ypred, ytest)


# In[36]:


ypred


# In[37]:


# Print the confusion matrix
confusion_matrix(ytest, ypred)


# In[38]:


# to check how probabilities are assigned
pypred = lrmodel.predict_proba(xtest_scaled)
pypred


# In[28]:


# to change the default threshold and to make it 70% and above
(pypred[:, 1] >= 0.7).astype(int)

# Or create a custom function to change the default threshold
class LRT(LogisticRegression):
    def predict(self, x, threshold=None):
        if threshold == None: # If no threshold is passed, simply use predict, where threshold is 0.5
            return LogisticRegression.predict(self, x)
        else:
            yscores = LogisticRegression.predict_proba(self, x)[:, 1]
            #print(yscores >= threshold)
            ypred_with_threshold = (yscores >= threshold).astype(int)

            return ypred_with_threshold
#     

#     

# ## <font color='chocolate'>**2. Random Forest**

# In[51]:


# Import RandomForestClassifier 
from sklearn.ensemble import RandomForestClassifier


# In[30]:


# Let's list the tunable hyperparameters for Random Forest algorithm
RandomForestClassifier().get_params()


# For random forests,
# 
# * The first hyperparameter to tune is n_estimators. We will try 100 and 200.
# 
# * The second one is max_features. Let's try - 'auto', 'sqrt', and 0.33.
# 
# * The third one is min_samples_leaf. Let's try - 1, 3, 5, 10

# In[57]:


rfmodel = RandomForestClassifier(n_estimators=100, 
                                 min_samples_leaf=5, 
                                 max_features='auto')
rfmodel.fit(xtrain, ytrain)

# predict on xtest
ypred = rfmodel.predict(xtest)

from sklearn.metrics import accuracy_score, confusion_matrix

print(accuracy_score(ypred, ytest),'\n')
print(confusion_matrix(ytest, ypred))


#     

# ## Cross Validation

# In[46]:


# import rquired libraries
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# if you have a imbalanced dataset, you can use stratifiedKFold
from sklearn.model_selection import StratifiedKFold


# ### For Logistic Regression

# In[55]:


# Set up a KFold cross-validation
kfold = KFold(n_splits=5)

# Use cross-validation to evaluate the model
lr_scores = cross_val_score(lrmodel, xtrain_scaled, ytrain, cv=kfold)

# Print the accuracy scores for each fold
print("Accuracy scores:", lr_scores)

# Print the mean accuracy and standard deviation of the model
print("Mean accuracy:", lr_scores.mean())
print("Standard deviation:", lr_scores.std())


# ### For Random Forest

# In[56]:


# Set up a KFold cross-validation
kfold = KFold(n_splits=5)

# Use cross-validation to evaluate the model
rf_scores = cross_val_score(rfmodel, xtrain_scaled, ytrain, cv=kfold)

# Print the accuracy scores for each fold
print("Accuracy scores:", rf_scores)

# Print the mean accuracy and standard deviation of the model
print("Mean accuracy:", rf_scores.mean())
print("Standard deviation:", rf_scores.std())


# ### Note:
# 
# 1. By using cross-validation, we can get a better estimate of the performance of the model than by using a single train-test split. This is because cross-validation uses all the data for training and testing, and averages the results over multiple iterations, which helps to reduce the impact of random variations in the data.
# <br><br>
# 2. **StratifiedKFold** is a variation of KFold that preserves the proportion of samples for each class in each fold. This is important when the target variable is imbalanced, i.e., when some classes have many more samples than others. By preserving the class proportions in each fold, StratifiedKFold ensures that each fold is representative of the overall dataset and helps to avoid overfitting or underfitting on specific classes.

# In[ ]:




