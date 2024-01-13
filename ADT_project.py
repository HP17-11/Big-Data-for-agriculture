#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from pymongo import MongoClient
client = MongoClient("localhost", 27017)


# In[3]:


db = client.Crop_Analysis


# In[4]:


collection = db.Cropdata
test = collection.find_one()
print(test)
print(collection)


# In[5]:


data = pd.DataFrame(list(collection.find()))
data


# In[6]:


df = data.drop(columns=['_id'])
df.head()


# In[7]:


for dirname, _, filenames in os.walk('C:/Users/kingh/Desktop/ADT_Project'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[8]:


df=pd.read_csv('C:/Users/kingh/Desktop/ADT_Project/Crop_recommendation.csv')
df.head()


# In[9]:


df.describe()


# In[10]:


df.dtypes


# In[11]:


sns.heatmap(df.isnull(),cmap="coolwarm")
plt.show()


# In[12]:


plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
sns.distplot(df['temperature'],color="purple",bins=15,hist_kws={'alpha':0.2})
plt.subplot(1, 2, 2)
sns.distplot(df['ph'],color="green",bins=15,hist_kws={'alpha':0.2})


# In[13]:


sns.countplot(y='label',data=df, palette="plasma_r")


# In[14]:


sns.pairplot(df, hue = 'label')


# In[15]:


sns.jointplot(x="rainfall",y="humidity",data=df[(df['temperature']<30) & (df['rainfall']>120)],hue="label")


# In[16]:


sns.jointplot(x="K",y="N",data=df[(df['N']>40)&(df['K']>40)],hue="label")


# In[17]:


sns.jointplot(x="K",y="humidity",data=df,hue='label',size=8,s=30,alpha=0.7)


# In[18]:


sns.boxplot(y='label',x='ph',data=df)


# In[19]:


sns.boxplot(y='label',x='P',data=df[df['rainfall']>150])


# In[20]:


sns.lineplot(data = df[(df['humidity']<65)], x = "K", y = "rainfall",hue="label")


# In[ ]:




