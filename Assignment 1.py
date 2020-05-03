#!/usr/bin/env python
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# # Assignment 1 - Introduction to Machine Learning

# For this assignment, you will be using the Breast Cancer Wisconsin (Diagnostic) Database to create a classifier that can help diagnose patients. First, read through the description of the dataset (below).

# In[97]:


import numpy as np
import pandas as pd
import seaborn as sbn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[4]:


cancer = load_breast_cancer()


# In[7]:


## print(cancer.DESCR)


# In[9]:


cancer.keys()


# The object returned by `load_breast_cancer()` is a scikit-learn Bunch object, which is similar to a dictionary.

# In[16]:


print(len(cancer['feature_names']))                                 ## Number of features


# In[17]:


print(cancer['feature_names'])                                      ## feature names


# In[20]:


cancer_data_frame = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[77]:


x = pd.DataFrame(cancer['data'] , columns = cancer['feature_names'])                                     
y = pd.DataFrame(cancer['target'] , columns = ['target'] ) 
z = m_n = pd.concat([x,y],axis=1) 

## Transform from Bunch to Dataframe
## pd.DataFrame(data=np.c_[cancer.data, cancer.target], columns=list(cancer.feature_names) + ['target'])


# In[90]:


values = z['target'].value_counts()
values.index = ['benign', 'malignant']
values


# In[80]:


sbn.set_style('whitegrid')
sbn.countplot(x = 'target', data = z , palette = 'RdBu_r')


# In[92]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


# In[99]:


knn = KNeighborsClassifier(n_neighbors = 1)                   

## 1-Nearest Neighbors Classification


# In[100]:


knn.fit(x_train, y_train)

##  Training the classifier using the training data


# In[101]:


knn.score(x_test, y_test)

## Accuracy of the classifier


# In[105]:


mean_values = x.mean().values.reshape(1, -1)
knn.predict(mean_values)

## Mean Values


# In[107]:


print(knn.predict(mean_values))

## Prediction of the Mean Values


# In[110]:


knn.predict(x_test)
len(knn.predict(x_test))
## Prediction of the Data Test


# In[ ]:





# In[ ]:





# In[ ]:




