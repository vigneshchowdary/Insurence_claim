#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


insurence=pd.read_csv(r"C:\Users\Vignesh Chowdary\OneDrive\Documents\Downloads\insurance2.csv")


# In[3]:


print(insurence.head())


# In[4]:


insurence.info(10)


# In[5]:


insurence.dtypes


# In[6]:


insurence.describe()


# In[7]:


insurence.isnull().sum()


# In[8]:


insurence.head()


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


x=insurence[['age','sex','bmi','region','charges','smoker']]
y=insurence['insuranceclaim']


# In[11]:


x.head()


# In[12]:


y.head()


# In[13]:


from sklearn.preprocessing import MinMaxScaler


# In[14]:


mms=MinMaxScaler()
x_scaled=mms.fit_transform(x)


# In[15]:


from sklearn.model_selection import train_test_split


# In[49]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.40,random_state=2)


# In[50]:


from sklearn.tree import DecisionTreeClassifier


# In[51]:


model=DecisionTreeClassifier()
model.fit(x_train,y_train)
pred=model.predict(x_test)


# In[52]:


x_train.shape


# In[53]:


y_train.shape


# In[54]:


pred


# In[55]:


result=x_test


# In[56]:


result['Actual'] =y_test
result['Predicted'] =pred


# In[57]:


result.head(15)


# In[58]:


model.fit(x,y)


# In[59]:


model.score(x,y)

