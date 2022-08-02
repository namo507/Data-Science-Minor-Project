#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Importing the File

# In[3]:


df1 = pd.read_csv('voice.csv')
df1.head()


# # Checking Null Values

# In[5]:


df1.isnull().sum()


# # Calculating stats of each column

# In[8]:


df1.describe()


# In[ ]:




