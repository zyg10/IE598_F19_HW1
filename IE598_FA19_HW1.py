#!/usr/bin/env python
# coding: utf-8

# In[1]:


#All of the code are copied from the book
from sklearn import datasets
iris = datasets.load_iris()
X_iris, y_iris = iris.data, iris.target
print(X_iris.shape, y_iris.shape)
print(X_iris[0], y_iris[0])


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing
 # Get dataset with only the first two attributes
x = X_iris[:, :2]
y = y_iris
# Split the dataset into a training and a testing set
    # Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=33)
print(X_train.shape, y_train.shape)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# print(X_train.shape, y_train.shape(112, 2) (112,))
#    # Standardize the features
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)


# In[3]:


import matplotlib.pyplot as plt
colors = ['red', 'greenyellow', 'blue']
for i in range(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')


# In[4]:



from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train, y_train)


# In[5]:


print(clf.coef_)


# In[6]:


print(clf.intercept_)


# In[ ]:





# In[7]:


import numpy as np
x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
#error in case Xs or xs
Xs = np.arange(x_min, x_max, 0.5)
fig, axes = plt.subplots(1, 3)
fig.set_size_inches(10, 6)
for i in [0, 1, 2]:
    axes[i].set_aspect('equal')
    axes[i].set_title('Class '+ str(i) + ' versus the rest')
    axes[i].set_xlabel('Sepal length')
    axes[i].set_ylabel('Sepal width')
    axes[i].set_xlim(x_min, x_max)
    axes[i].set_ylim(y_min, y_max)
    #error here need plt.
    plt.sca(axes[i])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.prism)
    ys = (-clf.intercept_[i] - Xs * clf.coef_[i, 0]) / clf.coef_[i, 1]
    plt.plot(Xs, ys)


# In[8]:


print( clf.predict(scaler.transform([[4.7, 3.1]])) )


# In[9]:


print( clf.decision_function(scaler.transform([[4.7, 3.1]])) )


# In[10]:


from sklearn import metrics
y_train_pred = clf.predict(X_train)
print( metrics.accuracy_score(y_train, y_train_pred) )


# In[11]:


y_pred = clf.predict(X_test)
print( metrics.accuracy_score(y_test, y_pred) )


# In[12]:


print( metrics.classification_report(y_test, y_pred, target_names=iris.target_names) )


# In[13]:


print( metrics.confusion_matrix(y_test, y_pred) )


# In[14]:


print("My name is Zhuoyuan Zhang")
print("My NetID is: zz10")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")


# In[ ]:





# In[ ]:




