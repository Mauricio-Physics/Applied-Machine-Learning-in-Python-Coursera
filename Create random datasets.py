#!/usr/bin/env python
# coding: utf-8

# ## Create Random Datasets

# In[6]:


from sklearn.datasets import make_classification, make_blobs
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_regression

import matplotlib.pyplot as plt


# In[7]:


## Synthetic dataset for simple regression

plt.figure()
plt.title('Sample regression problem with one input variable')
X_r, y_r = make_regression(n_samples=100, n_features=1, n_informative=1, bias = 150.0,noise = 30, random_state=0)
plt.scatter(X_r, y_r, marker= 'o', s=50)
plt.show()

## https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html


# In[8]:


## Synthetic dataset for classification (binary) 

cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

plt.figure()
plt.title('Sample binary classification problem with two informative features')
X_c, y_c = make_classification(n_samples = 100, n_features=2, n_redundant=0, n_informative=2, 
                               n_clusters_per_class=1, flip_y = 0.1, class_sep = 0.5, random_state=0)
plt.scatter(X_c[:, 0], X_c[:, 1], c=y_c, marker= '*', s=50, cmap=cmap_bold)
plt.show()

## https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html


# In[9]:


# More difficult synthetic dataset for classification (binary) with classes that are not linearly separable

X_d, y_d = make_blobs(n_samples = 100, n_features = 2, centers = 8,cluster_std = 1.3, random_state = 4)
y_d = y_d % 2
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_d[:,0], X_d[:,1], c=y_d,marker= 'o', s=50, cmap=cmap_bold)
plt.show()

## https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html


# In[ ]:




