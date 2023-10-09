#!/usr/bin/env python
# coding: utf-8

# In[228]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

titanic_data = pd.read_csv('C:\\Users\\ms\\PycharmProjects\\pythonProject1\\set.csv')



titanic_data.isnull().sum()



titanic_data.info()




titanic_data.shape




titanic_data = titanic_data.drop(columns='Cabin', axis=1)





titanic_data.head()




titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)




titanic_data['Fare'].fillna(titanic_data['Fare'].mean(), inplace=True)








# In[229]:


titanic_data.info()


# In[230]:


titanic_data.isnull().sum()


# In[231]:


titanic_data.describe()


# In[232]:


titanic_data['Survived'].value_counts()


# In[233]:


sns.set()


# In[234]:


sns.countplot(x='Survived', data=titanic_data)


# In[235]:


sns.countplot(x='Sex', data=titanic_data)


# In[236]:


sns.countplot(x='Pclass', data=titanic_data)


# In[237]:


sns.countplot(x='Pclass', hue='Survived', data=titanic_data)


# In[238]:


titanic_data['Embarked'].value_counts() 



# In[239]:


titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)


# In[240]:


titanic_data.head()


# In[241]:


X=titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'], axis=1)


# In[242]:


print(X)


# In[243]:


Y=titanic_data['Survived']


# In[244]:


print(Y)


# In[245]:


X_train, X_test , Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=4)


# In[246]:


model=LogisticRegression()


# 

# In[247]:


model.fit(X_train,Y_train)


# In[248]:


X_train_prediction=model.predict(X_train)


# In[249]:


print(X_train_prediction)


# In[250]:


training_accuracy=accuracy_score(Y_train, X_train_prediction)


# In[253]:


print("the accuracy is ", training_accuracy)


# In[252]:


X_test_prediction=model.predict(X_test)


# In[254]:


model.predict(X_test)


# In[285]:


import warnings
warnings.filterwarnings("ignore")

new_passenger_features = pd.DataFrame({
    'Pclass': [3],  
    'Sex': [0],     
    'Age': [titanic_data['Age'].mean()],  
    'Embarked': [titanic_data['Embarked'].mean()],
    'Fare': [titanic_data['Fare'].mean()], 
    'Parch': [titanic_data['Parch'].mean()],
    'SibSp': [titanic_data['SibSp'].mean()]
}, columns=X.columns)

# Make predictions using the trained model
prediction = model.predict(new_passenger_features)

if prediction == 0:
    print("Not survived")
else:
    print("Survived")
    


# In[ ]:




