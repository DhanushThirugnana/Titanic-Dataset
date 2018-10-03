
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math

titanic_data=pd.read_csv("C:/Users/dhanu/Desktop/Tree/UTD/Machine Learning/Programs/Logistic reg/train.csv")
titanic_data.head(10)


# In[7]:


print("No of passengers = ",len(titanic_data.index))


# In[8]:


#Analyze data
sns.countplot(x='Survived',data=titanic_data)


# In[9]:


sns.countplot(x='Survived',hue='Sex',data=titanic_data)


# In[10]:


sns.countplot(x="Survived",hue='Pclass',data=titanic_data)


# In[13]:


titanic_data["Age"].plot.hist()


# In[15]:


## Data Cleaning
titanic_data.isnull()


# In[16]:


titanic_data.isnull().sum()


# In[26]:


sns.heatmap(titanic_data.isnull(),yticklabels=False)


# In[21]:


titanic_data.drop("Cabin",axis=1,inplace=True)


# In[23]:


titanic_data.dropna(inplace=True)


# In[25]:


sns.heatmap(titanic_data.isnull(), yticklabels=False)


# In[27]:


titanic_data.isnull().sum()


# In[32]:


sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)


# In[34]:


sex.head(5)


# In[36]:


embark=pd.get_dummies(titanic_data["Embarked"],drop_first=True)
embark.head(5)


# In[37]:


passengers=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
passengers.head(5)


# In[38]:


titanic_data=pd.concat([titanic_data,sex,passengers,embark],axis=1)


# In[39]:


titanic_data.head(5)


# In[40]:


titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass'],axis=1,inplace=True)


# In[43]:


titanic_data.head(5)


# In[45]:


###train data
x=titanic_data.drop('Survived',axis=1)
y=titanic_data['Survived']


# In[47]:


from sklearn.model_selection import train_test_split


# In[52]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=1)


# In[50]:


from sklearn.linear_model import LogisticRegression


# In[51]:


logReg=LogisticRegression()


# In[53]:


logReg.fit(x_train,y_train)


# In[55]:


predictions=logReg.predict(x_test)


# In[56]:


from sklearn.metrics import classification_report


# In[57]:


classification_report(y_test,predictions)


# In[59]:


from sklearn.metrics import confusion_matrix


# In[60]:


confusion_matrix(y_test,predictions)


# In[61]:


from sklearn.metrics import accuracy_score


# In[62]:


accuracy_score(y_test,predictions)

