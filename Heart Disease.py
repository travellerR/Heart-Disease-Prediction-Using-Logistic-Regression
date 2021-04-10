#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('heart.csv')


# In[2]:


df = pd.read_csv('heart.csv')


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[8]:


print(df.info())


# In[12]:


import seaborn as sns
plt.figure(figsize = (20, 10))
sns.heatmap(df.corr(), annot = True, cmap = 'terrain')


# In[14]:


df.hist(figsize = (12,12), layout = (5,3));


# In[15]:


df.plot(kind = 'box', subplots = True, layout = (5,3), figsize = (12,12))
plt.show()


# In[16]:


sns.catplot(data=df, x='sex', y='age',  hue='target', palette='husl')


# In[17]:


sns.barplot(data=df, x='sex', y='chol', hue='target', palette='spring')


# In[18]:


df['sex'].value_counts()


# In[19]:


df['target'].value_counts()


# In[20]:


df['thal'].value_counts()


# In[21]:


sns.countplot(x='sex', data=df, palette='husl', hue='target')


# In[22]:


sns.countplot(x='target',palette='BuGn', data=df)


# In[23]:


sns.countplot(x='ca',hue='target',data=df)


# In[24]:


df['ca'].value_counts()


# In[25]:


sns.countplot(x='thal',data=df, hue='target', palette='BuPu' )


# In[26]:


sns.countplot(x='thal', hue='sex',data=df, palette='terrain')


# In[27]:


df['cp'].value_counts()


# In[28]:


sns.countplot(x='cp' ,hue='target', data=df, palette='rocket')


# In[29]:


sns.countplot(x='cp', hue='sex',data=df, palette='BrBG')


# In[30]:


sns.boxplot(x='sex', y='chol', hue='target', palette='seismic', data=df)


# In[31]:


sns.barplot(x='sex', y='cp', hue='target',data=df, palette='cividis')


# In[32]:


sns.barplot(x='sex', y='thal', data=df, hue='target', palette='nipy_spectral')


# In[33]:


sns.barplot(x='target', y='ca', hue='sex', data=df, palette='mako')


# In[34]:


sns.barplot(x='sex', y='oldpeak', hue='target', palette='rainbow', data=df)


# In[35]:


df['fbs'].value_counts()


# In[36]:


sns.barplot(x='fbs', y='chol', hue='target', data=df,palette='plasma' )


# In[37]:


sns.barplot(x='sex',y='target', hue='fbs',data=df)


# In[38]:


gen = pd.crosstab(df['sex'], df['target'])
print(gen)


# In[39]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()  
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
df[columns_to_scale] = StandardScaler.fit_transform(df[columns_to_scale])


# In[40]:


df.head()


# In[41]:


X= df.drop(['target'], axis=1)
y= df['target']


# In[42]:


X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)


# In[43]:


print('X_train-', X_train.size)
print('X_test-',X_test.size)
print('y_train-', y_train.size)
print('y_test-', y_test.size)


# In[44]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)


# In[45]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,prediction1)
cm


# In[46]:


TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]
print('Testing Accuracy:',(TP+TN)/(TP+TN+FN+FP))


# In[47]:


from sklearn.metrics import classification_report
print(classification_report(y_test, prediction1))


# In[ ]:




