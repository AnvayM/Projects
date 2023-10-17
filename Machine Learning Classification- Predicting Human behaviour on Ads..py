#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression Project 
# 
# In this project I have worked fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. I have created a model that will predict whether or not they will click on an ad based off the features of that user.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# 

# In[2]:


ad_data = pd.read_csv("D:\\Practice folder\\Python\\Py-DS-ML-Bootcamp-master\\Refactored_Py_DS_ML_Bootcamp-master\\13-Logistic-Regression\\advertising.csv")


# In[3]:


ad_data


# In[4]:


ad_data.info()


# In[5]:


ad_data.describe()


# ## Exploratory Data Analysis
# 

# In[6]:


#Understanding which age people uses the internet the most. 
sns.displot(data= ad_data, x= 'Age', kind='hist' )
plt.figure(figsize=(5,5))


# In[7]:


#Understanding the relationship between Age and Area Income.
sns.jointplot(data= ad_data, x ='Age',y ='Area Income')


# In[8]:


#Understanding the relationship between Age and Daily Time Spent on Site.
sns.jointplot(data=ad_data, x= 'Age', y= "Daily Time Spent on Site",kind='kde',palette='muted')
plt.grid()


# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# In[9]:


#Understanding teh relationship between the Daily Time Spent on Site and Daily Internet Usage.
sns.jointplot(data=ad_data, x= 'Daily Time Spent on Site', y="Daily Internet Usage", palette='husl')


# In[10]:


#Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature
y=sns.PairGrid(ad_data, hue='Clicked on Ad')
y.map_upper(sns.scatterplot)
y.map_lower(sns.scatterplot)
y.map_diag(sns.histplot)


# In[11]:


corr = ad_data.corr(numeric_only=True)


# In[12]:


#Checking the missing values
sns.heatmap(corr, yticklabels=True, cbar=True, cmap='viridis', annot=True)


# In[13]:


ad_data["months"] = pd.to_datetime(ad_data['Timestamp']).dt.month


# In[14]:


month = pd.get_dummies(ad_data['months'], drop_first=True)


# In[15]:


ad_data = pd.concat([ad_data,month], axis=1)


# In[16]:


#removing unwanted columns
ad_data.drop(['Country',"City","Timestamp","Ad Topic Line"], axis=1, inplace=True)


# In[17]:


#Checking for outliers
plt.figure(figsize=(10,5))
sns.boxplot(data =ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male', 'Clicked on Ad']])


# In[18]:


#Converting all the num col name into string in order to avoid error in .fit()
ad_data.columns = ad_data.columns.astype(str)


# # Logistic Regression
# 
# 

# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# In[20]:


X = ad_data.drop('Clicked on Ad', axis=1)
y = ad_data['Clicked on Ad']


# ## Split the data into training set and testing set using train_test_split

# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=101)


# In[22]:


pipe = make_pipeline(StandardScaler(),LogisticRegression())


# In[23]:


pipe.fit(X_train,y_train)


# ## Predictions and Evaluations
# 

# In[24]:


prediction = pipe.predict(X_test)


# In[25]:


from sklearn.metrics import classification_report


# In[26]:


print(classification_report(y_test,prediction))

