#!/usr/bin/env python
# coding: utf-8

# # ROCK_OR_MINE

# There is war is going on between two countries submarine of the country is going under the water to another country and enemy country planted some mines in the oceans mine are nothing but explosive that explodes when some object comes in contact with it and there can be rocks in the ocean so submarine needs to predict whether it is crossing mine or rock our job is to make a system that can predict whether the object beneath the submarine is a mine or a rock so how this is done is submarine uses sonar signal that sends sound and receives switchbacks so this signal in the processed to detect whether the object is a mine or it's just a rock in the ocean to predict the rock and mine we use some types of algorithms like decision tree, KNN, Logistic Regression, Random Forest and SVM

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv('ROCK_OR_MINE.csv',header=None)
df


# In[3]:


df.shape


# In[4]:


df.describe()


# In[5]:


df.value_counts()


# In[6]:


df[60].value_counts()


# In[7]:


df.groupby(60).mean()


# In[8]:


#seperating data and labels
x=df.drop(columns=60 , axis=1)
y=df[60]


# In[9]:


print(x,y)


# In[10]:


x_train , x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)
print(x.shape,x_train.shape,x_test.shape,y.shape,y_train.shape,y_test.shape)


# In[11]:


print(x_train,y_train)


# Model Training --> Logistic Regression

# In[12]:


model=LogisticRegression()


# In[13]:


model.fit(x_train,y_train)


# Model Evaluation

# In[14]:


x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print('Accuracy on training data :',training_data_accuracy)


# In[15]:


x_test_prediction=model.predict(x_test)
testing_data_accuracy=accuracy_score(x_test_prediction,y_test)
print('Accuracy on testing data:',testing_data_accuracy)


# Making a Predictive System

# In[16]:


data=(0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.1460,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.8230,0.9173,0.9975,0.9911,0.8240,0.6498,0.5980,0.4862,0.3150,0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.2930,0.2925,0.3998,0.3660,0.3172,0.4609,0.4374,0.1820,0.3376,0.6202,0.4448,0.1863,0.1420,0.0589,0.0576,0.0672,0.0269,0.0245,0.0190,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055)
d1=np.asarray(data)
reshape_d1=d1.reshape(1,-1)
print(reshape_d1)


# In[17]:


prediction=model.predict(reshape_d1)
print(prediction)


# In[18]:


if (prediction[0]=='R'):
    print('Object is Rock')
else:
    print('Object is Mine')

