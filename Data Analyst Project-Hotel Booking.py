#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df= pd.read_csv("D:\\DataScience\\Projects\\Data Analysis\\Hotel Booking\\hotel_booking.csv")
df


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


#Converting reservation_status _date into datetime format
df['reservation_status_date']=pd.to_datetime(df['reservation_status_date'])


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


#Getting details about only catagorical columns.
df.describe(include='object')


# In[11]:


#Checking the names of catagorical cols.
df.describe(include='object').columns


# In[12]:


# Seeing the data present in the cata cols
for col in df.describe(include='object').columns:
    print(df[col].unique())
    print("-"*50)


# In[13]:


df.isnull().sum()


# In[14]:


df.drop(['country','agent'], axis=1, inplace = True)


# In[15]:


df.drop(['name','email','phone-number','credit_card'], axis=1, inplace=True)


# In[16]:


df.info()


# In[17]:


df.isnull().sum()


# In[18]:


df.describe()


# In[19]:


sns.boxplot(df,y='adr')


# In[20]:


df = df[df['adr']<5000] 


# In[21]:


df.describe()


# In[22]:


df['adr'].plot(kind='box')


# # Data Analysis and Visualization

# In[23]:


cancelled_perc = df['is_canceled'].value_counts(normalize=True)
cancelled_perc


# In[24]:


plt.figure(figsize=(5,5))
plt.title("is_cancelled")
plt.bar(["cancelled","not_cancelled"],df['is_canceled'].value_counts())


# In[25]:


#Now we will check in which hotel the cancellation.
cancellation_stats = sns.countplot(data=df, x='hotel', hue='is_canceled', palette='bright')
plt.legend(['cancelled',"not_cancelled"])
plt.xlabel('Hotels')
plt.ylabel("number of reservation")
plt.show()


# In[26]:


#Now we are checking in resort hotel what is the % of conccellation
resort_hotel = df[df['hotel']=='Resort Hotel']
city_hotel = df[df['hotel']=='City Hotel']


# In[27]:


city_hotel['is_canceled'].value_counts(normalize=True)


# In[28]:


resort_hotel['is_canceled'].value_counts(normalize=True)


# In[29]:


resort_hotel= resort_hotel.groupby('reservation_status_date')[['adr']].mean()
city_hotel = city_hotel.groupby('reservation_status_date')[['adr']].mean()


# In[30]:


plt.figure(figsize=(30,10))
plt.plot(resort_hotel.index,resort_hotel['adr'], label = 'Resort Hotel')
plt.plot(city_hotel.index,city_hotel['adr'], label = 'city Hotel')
plt.legend(fontsize=20)


# In[31]:


#In whcih month the reservation happens the most.
plt.figure(figsize=(10,10))
df['months']=df['reservation_status_date'].dt.month
month_status = sns.countplot(data=df, x='months', hue='is_canceled')
plt.legend(['not_cancelled','is_cancelled'])


# In[32]:


#Comparing the price vs per month cancellation rate
plt.figure(figsize=(10,10))
plt.title("adr per month", fontsize=10)
plt.bar('months','adr', data=df[df['is_canceled']==1].groupby('months')[['adr']].sum().reset_index())
plt.show()


# In[33]:


#Checking through which source the people are coming to the hotel.
df['market_segment'].value_counts(normalize=True)


# In[34]:


plt.figure(figsize=(12,10))
sns.countplot(x='market_segment', data=df)


# In[ ]:




