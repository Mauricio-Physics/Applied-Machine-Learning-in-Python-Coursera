#!/usr/bin/env python
# coding: utf-8

# In[13]:


from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from adspy_shared_utilities import plot_class_regions_for_classifier
from sklearn.datasets import make_classification, make_blobs
from sklearn.datasets import load_breast_cancer


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ### Multi-class classification with linear models

# In[2]:


fruits = pd.read_table('C:/Users/MAURICIO/Desktop/fruit_data_with_colors.txt')

X_fruits_2d = fruits[['height', 'width']]
y_fruits_2d = fruits['fruit_label']


# In[3]:


X_train, X_test, y_train, y_test = train_test_split(X_fruits_2d, y_fruits_2d, random_state = 0)

clf = LinearSVC(C=5, random_state = 67).fit(X_train, y_train)
print('Coefficients:\n', clf.coef_)
print('Intercepts:\n', clf.intercept_)


# In[5]:


plt.figure(figsize=(6,6))
colors = ['r', 'g', 'b', 'y']
cmap_fruits = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])


plt.scatter(X_fruits_2d[['height']], X_fruits_2d[['width']], cmap=cmap_fruits, edgecolor = 'black', alpha=.7)

x_0_range = np.linspace(-10, 15)

for w, b, color in zip(clf.coef_, clf.intercept_, ['r', 'g', 'b', 'y']):
    # Since class prediction with a linear model uses the formula y = w_0 x_0 + w_1 x_1 + b, 
    # and the decision boundary is defined as being all points with y = 0, to plot x_1 as a 
    # function of x_0 we just solve w_0 x_0 + w_1 x_1 + b = 0 for x_1:
    plt.plot(x_0_range, -(x_0_range * w[0] + b) / w[1], c=color, alpha=.8)
    
plt.legend(fruits['fruit_name'].unique())
plt.xlabel('height')
plt.ylabel('width')
plt.xlim(-2, 12)
plt.ylim(-2, 15)
plt.show()


# ## Kernelized Support Vector Machines

# In[9]:


cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

X_d, y_d = make_blobs(n_samples = 100, n_features = 2, centers = 8,cluster_std = 1.3, random_state = 4)
y_d = y_d % 2
plt.figure()
plt.title('Sample binary classification problem with non-linearly separable classes')
plt.scatter(X_d[:,0], X_d[:,1], c=y_d,marker= 'o', s=50, cmap=cmap_bold)
plt.show()


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X_d, y_d, random_state = 0)

# The default SVC kernel is radial basis function (RBF)

plot_class_regions_for_classifier(SVC().fit(X_train, y_train),X_train, y_train, None, None,
                                  'Support Vector Classifier: RBF kernel')

# Compare decision boundries with polynomial kernel, degree = 3

plot_class_regions_for_classifier(SVC(kernel = 'poly', degree = 3).fit(X_train, y_train), X_train,y_train, None, None,
                                 'Support Vector Classifier: Polynomial kernel, degree = 3')


# #### Support Vector Machine with RBF kernel: gamma parameter

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X_d, y_d, random_state = 0)
fig, subaxes = plt.subplots(3, 1, figsize=(4, 11))

for this_gamma, subplot in zip([0.01, 1.0, 5.0], subaxes):
    clf = SVC(kernel = 'rbf', gamma=this_gamma).fit(X_train, y_train)
    title = 'Support Vector Classifier: \nRBF kernel, gamma = {:.2f}'.format(this_gamma)
    plot_class_regions_for_classifier_subplot(clf, X_train, y_train,
                                             None, None, title, subplot)
    plt.tight_layout()


# ### Application of SVMs to a real dataset: unnormalized data

# In[14]:


cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)
                                                   

clf = SVC(C=10).fit(X_train, y_train)
print('Breast cancer dataset (unnormalized features)')
print('Accuracy of RBF-kernel SVC on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of RBF-kernel SVC on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# ### Application of SVMs to a real dataset: normalized data with feature preprocessing using minmax scaling
# 

# In[16]:


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SVC(C=10).fit(X_train_scaled, y_train)
print('Breast cancer dataset (normalized with MinMax scaling)')
print('RBF-kernel SVC (with MinMax scaling) training set accuracy: {:.2f}'
     .format(clf.score(X_train_scaled, y_train)))
print('RBF-kernel SVC (with MinMax scaling) test set accuracy: {:.2f}'
     .format(clf.score(X_test_scaled, y_test)))


# In[ ]:




