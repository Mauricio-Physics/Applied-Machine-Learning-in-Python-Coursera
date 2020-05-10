#!/usr/bin/env python
# coding: utf-8

# In[14]:


from sklearn.linear_model import LogisticRegression
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot
from sklearn.datasets import load_breast_cancer


import matplotlib.pyplot as plt


# ## Linear models for classification

# ### Logistic regression

# In[15]:


cmap_bold = ListedColormap(['#FFFF00', '#00FF00', '#0000FF','#000000'])

plt.figure()
plt.title('Sample binary classification problem with two informative features')
X_c, y_c = make_classification(n_samples = 100, n_features=2, n_redundant=0, n_informative=2, 
                               n_clusters_per_class=1, flip_y = 0.1, class_sep = 0.5, random_state=0)
plt.scatter(X_c[:, 0], X_c[:, 1], c=y_c, marker= '*', s=50, cmap=cmap_bold)
plt.show()


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X_c, y_c,random_state = 0)

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
c_1 = LogisticRegression(C=0.1).fit(X_train, y_train)
title = 'Logistic regression, simple synthetic dataset C = {:.3f}'.format(0.1)
plot_class_regions_for_classifier_subplot(c_1, X_train, y_train, None, None, title, subaxes)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(c_1.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(c_1.score(X_test, y_test)))


# In[17]:


fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
c_2 = LogisticRegression(C=1).fit(X_train, y_train)
title = 'Logistic regression, simple synthetic dataset C = {:.3f}'.format(1.0)
plot_class_regions_for_classifier_subplot(c_2, X_train, y_train, None, None, title, subaxes)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(c_2.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(c_2.score(X_test, y_test)))


# In[18]:


fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
c_3 = LogisticRegression(C=10).fit(X_train, y_train)
title = 'Logistic regression, simple synthetic dataset C = {:.3f}'.format(10)
plot_class_regions_for_classifier_subplot(c_3, X_train, y_train, None, None, title, subaxes)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(c_3.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(c_3.score(X_test, y_test)))


# ### Support Vector Machines

# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, random_state = 0)

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
c_1 = 0.1
clf = SVC(kernel = 'linear', C=c_1).fit(X_train, y_train)
title = 'Linear SVC, C = {:.2f}'.format(c_1)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, random_state = 0)

fig, subaxes = plt.subplots(1, 1, figsize=(7, 5))
c_2 = 10.0
clf = SVC(kernel = 'linear', C=c_2).fit(X_train, y_train)
title = 'Linear SVC, C = {:.2f}'.format(c_2)
plot_class_regions_for_classifier_subplot(clf, X_train, y_train, None, None, title, subaxes)


# In[21]:


cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, random_state = 0)

clf = LinearSVC().fit(X_train, y_train)
print('Breast cancer dataset')
print('Accuracy of Linear SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Linear SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))


# In[ ]:




