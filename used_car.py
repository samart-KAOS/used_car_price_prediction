#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# In[2]:


os.chdir (r'C:\Users\samart\Documents\clg internship')


# In[3]:


ds = pd.read_csv("car.csv")
ds.head(5)


# In[4]:


data=ds.drop(["Year"],axis=1)
data=data.drop(["Owner"],axis=1)
data=data.drop(["Car_Name"],axis=1)
x=data.drop(["Selling_Price"],axis=1)
y=data.pop('Selling_Price')


# In[5]:


x.rename(columns={'Present_Price':'Cost'}, inplace = True)


# In[6]:


y


# In[7]:


xtrain, xtest, ytrain, ytest= train_test_split(x,y,test_size=0.3,random_state=10)
xtest


# In[8]:


xtrain=pd.get_dummies(xtrain,columns=["Fuel_Type","Seller_Type","Transmission"], drop_first=True)


# In[9]:


xtest=pd.get_dummies(xtest,columns=["Fuel_Type","Seller_Type","Transmission"], drop_first=True)


# In[10]:


missing_cols = set (xtrain.columns) - set (xtest.columns)
for col in missing_cols:
    xtest[col] = 0
xtest = xtest[xtrain.columns]


# In[11]:


ss=StandardScaler()
ss.fit(xtrain)
xtrain=ss.transform(xtrain)
xtest=ss.transform(xtest)


# In[12]:


rf=RandomForestRegressor(n_estimators =100)
rf.fit(xtrain,ytrain)
pred=rf.predict(xtest)
r2_score(ytest,pred)


# In[13]:


plt.scatter(ytest,pred)


# In[14]:


pred


# In[ ]:




