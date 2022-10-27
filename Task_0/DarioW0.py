#!/usr/bin/env python
# coding: utf-8

# # Week 1 Hand-In
# 
# Learning based on linear regression (I think)

# In[239]:


import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import svm, linear_model
import sklearn
from sklearn import neighbors
import pandas as pd


# ## Generate sample data
# 
# Import the sample data as numpy array. Reshape the data to fit d=10
# and n= to whatever the data set says. Extract y to make prediction with
# sklearn or something?

# In[240]:


train_data = np.genfromtxt('train.csv', delimiter=',')
train_data=np.delete(train_data,0,0)
train_data=np.delete(train_data,0,1)
y = train_data[:,0]
train_data=np.delete(train_data,0,1)

test_data = np.genfromtxt('test.csv', delimiter=',')
ind = test_data[:,0]
test_data = np.delete(test_data,0,0)
test_data = np.delete(test_data,0,1)


# Now lets try to fit the data to the target using Support Vector Classification (SVC).
# For this we use the included svc command in skilearn.

# In[241]:


classifiers = [
    #svm.SVR(kernel='linear', gamma='auto',tol=1e-9),
    #linear_model.SGDRegressor(),
    linear_model.BayesianRidge(tol=1e-16, n_iter=512),
    #linear_model.LassoLars(max_iter=500),
    linear_model.ARDRegression(tol=1e-8, n_iter=600),
    #linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]

title = ["Id", "y"]
temp = np.zeros([2000,2])
temp[:,0]= ind[1:]


# lets try to delete outliers from the dataset in order to have a better estimate of the overall data.
# 

# In[242]:


data_mean, data_std = np.mean(train_data), np.std(train_data)

lof = sklearn.neighbors.LocalOutlierFactor(n_neighbors= 60)
yhat = lof.fit_predict(train_data)

mask = yhat != -1
train_data, y = train_data[mask,:], y[mask]


# In[243]:


for item in classifiers:
    clf = item
    clf.fit(train_data,y)
    predict_train = clf.predict(train_data)
    predict_test = clf.predict(test_data)
    #print(clf.score(train_data,predict))
    print("Using %(n)s we have an Absolute Error of %(s)s on the TRAINING DATA" % {'n': item, 's': np.linalg.norm(y-predict_train)}, "\n")
    print("The Score for the test data is %s" % {'s': clf.score(test_data,predict_test)}, "\n")

    temp[:,1] = predict_test
    result = pd.DataFrame(temp, columns=title)
    print(result)
    result.to_csv("REEE%s.csv" %item, header=True, index = False)


# 7.183621407620719e-11
# 6.677525227032642e-11
# 4.802438449647887e-11
# 4.802438449647887e-11
# 
# 2.0998834488939323e-11 n_neighbour = 20
# 1.5426904780026072e-11 n_neighbour = 50
# 1.3229044701006721e-11 n_neighbour = 60
# 
# 
# 
# 6= 3.290098950485172e-11
# 8= 3.290098950485172e-11

# In[244]:


print(np.mean(train_data))
print(np.std(train_data))

