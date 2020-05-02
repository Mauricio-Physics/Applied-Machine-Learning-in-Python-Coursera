#!/usr/bin/env python
# coding: utf-8

# In[38]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from matplotlib import cm
from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from adspy_shared_utilities import plot_fruit_knn


# In[39]:


fruits = pd.read_table('C:/Users/MAURICIO/Desktop/fruit_data_with_colors.txt')


# In[40]:


fruits.head()


# In[41]:


lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   
lookup_fruit_name

# create a mapping from fruit label value to fruit name to make results easier to interpret


# In[42]:


x = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


# In[43]:


print(len(x_train)/len(x),len(y_train)/len(y))


# In[44]:


cmap = cm.get_cmap('gnuplot')
scatter = scatter_matrix(x_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)

# plotting a scatter matrix


# In[29]:


fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(x_train['width'], x_train['height'], x_train['color_score'], c = y_train, marker = 'o', s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()

# plotting a 3D scatter plot


# In[45]:


knn = KNeighborsClassifier(n_neighbors = 4)                   ## K-Nearest Neighbors Classification


# In[48]:


plot_fruit_knn(x_train, y_train, 1, 'uniform')                ## we choose 4 nearest neighbors


# In[47]:


plot_fruit_knn(x_train, y_train, 4, 'uniform')             ## we choose 4 nearest neighbors


# In[49]:


knn.fit(x_train, y_train)

##  Training the classifier using the training data 


# In[51]:


knn.score(x_test, y_test)

## Accuracy of the classifier


# In[58]:


fruit_prediction = knn.predict([[7.1, 5.5, 180,0.72]])
lookup_fruit_name[fruit_prediction[0]]

# first example predicting a label


# In[63]:


k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    scores.append(knn.score(x_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);

## Classification accuracy vs 'k' parameter


# In[67]:


t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1-s)
        knn.fit(x_train, y_train)
        scores.append(knn.score(x_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('Accuracy');


## Training set proportion vs Accuracy   (k=5)


# In[ ]:




