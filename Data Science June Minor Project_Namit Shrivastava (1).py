#!/usr/bin/env python
# coding: utf-8

# In[138]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Importing the File

# In[139]:


df1 = pd.read_csv('voice.csv')
df1.head()


# # Checking Null Values

# In[140]:


df1.isnull().sum()


# # Calculating stats of each column

# In[141]:


df1.describe()


# # Percentage Distribution Pie Chart 

# In[142]:


df1['label'].value_counts()


# In[143]:


#Since the count is equally distributed hence, 50% for both
import matplotlib.pyplot as plt

t = [1584,1584]
labels = ['Male', 'Female']
colors = ['tab:blue', 'tab:red']

fig, ax = plt.subplots()
ax.pie(t, labels = labels, colors = colors, autopct='%.0f%%')
ax.set_title('Percentage Distribution of Label')
plt.show()


# # Training and testing data Split

# In[144]:


x = df1[['meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt', 'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx']]
y = df1['label']
print(type(x))
print(type(y))
print(x.shape)
print(y.shape)


# In[145]:


from sklearn.model_selection import train_test_split


# In[146]:


x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Decision Tree Classifier

# In[147]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[148]:


def gen_metrics(ytest,ypred):
    cm = confusion_matrix(ytest,ypred)
    print('confusion matrix\n',cm)
    print('Classification report\n',classification_report(ytest,ypred))
    print('Acc Score', accuracy_score(ytest,ypred))


# In[149]:


from sklearn.tree import DecisionTreeClassifier


# In[150]:


m1 = DecisionTreeClassifier(criterion='gini', max_depth=10,min_samples_split=10)
m1.fit(x_train,y_train)


# In[151]:


#Accuracy
print('Training Score',m1.score(x_train,y_train))
print('Testing Score',m1.score(x_test,y_test))


# In[152]:


ypred_m1 = m1.predict(x_test)
print(ypred_m1)


# In[153]:


gen_metrics(y_test,ypred_m1)


# # Random Forest Classifier

# In[154]:


from sklearn.ensemble import RandomForestClassifier


# In[155]:


m2 = RandomForestClassifier(n_estimators=50,criterion='entropy',max_depth=6,min_samples_split=12)
m2.fit(x_train,y_train)


# In[156]:


print('Training Score',m2.score(x_train,y_train))
print('Testing Score',m2.score(x_test,y_test))


# In[157]:


ypred_m2 = m2.predict(x_test)
print(ypred_m2)


# In[158]:


gen_metrics(y_test,ypred_m2)


# # KNN Classifier

# In[159]:


from sklearn.neighbors import KNeighborsClassifier


# In[160]:


m3 = KNeighborsClassifier()
m3.fit(x_train,y_train)


# In[161]:


#Accuracy
print('Training Score',m3.score(x_train,y_train))
print('Testing Score',m3.score(x_test,y_test))


# In[162]:


ypred_m3 = m3.predict(x_test)
print(ypred_m3)


# In[180]:


gen_metrics(y_test,ypred_m3)


# # Logistic Regression

# In[165]:


from sklearn.linear_model import LogisticRegression


# In[166]:


m4 = LogisticRegression(max_iter=10000)
m4.fit(x_train,y_train)


# In[167]:


#Accuracy
print('Training Score',m4.score(x_train,y_train))
print('Testing Score',m4.score(x_test,y_test))


# In[168]:


ypred_m4 = m4.predict(x_test)
print(ypred_m4)


# In[181]:


gen_metrics(y_test,ypred_m4)


# In[171]:


m = m4.coef_
c = m4.intercept_
print('Coefficient',m)
print('Intercept or constant',c)


# In[172]:


print('Testing Score',m4.score(x_test,y_test))
print('Accuracy Score',accuracy_score(y_test,ypred_m4))


# In[173]:


def sigmoid(X,m,c):
    logit = 1/(1 + np.exp(-(m*X+c)))
    print(logit)


# In[174]:


sigmoid (0.077316,m,c) #example using any value taken as X


# # SVM Classifier

# In[175]:


from sklearn.svm import SVC


# In[176]:


m5 = SVC()
m5.fit(x_train,y_train)


# In[177]:


#Accuracy
print('Training Score',m5.score(x_train,y_train))
print('Testing Score',m5.score(x_test,y_test))


# In[178]:


ypred_m5 = m5.predict(x_test)
print(ypred_m5)


# In[179]:


gen_metrics(y_test,ypred_m5)


# # CONCLUSION: The Accuracy Score of each model is as follows:
# 
# 
# a. Decision Tree Classifier : Acc Score 0.9605678233438486<br>
# b. Random Forest Classifier : Acc Score 0.9826498422712934<br>
# c. KNN Classifier : Acc Score 0.7003154574132492<br>
# d. Logistic Regression : Acc Score 0.9100946372239748<br>
# e. SVM Classifier : Acc Score 0.6719242902208202<br>
# 
# Hence the model with the highest accuracy is Random Forest Classifier with an accuracy score of 0.9826498422712934
