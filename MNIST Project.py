#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns


# In[4]:


(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()


# In[5]:


len(x_train)


# In[6]:


x_train[0].shape


# In[7]:


plt.matshow(x_train[0])


# In[8]:


x_train.shape


# In[9]:


x_train =x_train/255
x_test = x_test/255


# In[10]:


x_train_flattened = x_train.reshape(len(x_train),28*28)
x_test_flattened = x_test.reshape(len(x_test),28*28)


# In[11]:


x_test_flattened.shape


# In[12]:


x_train_flattened.shape


# In[13]:


model= keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation="relu"),
    keras.layers.Dense(10,activation="sigmoid")
])

model.compile(
    optimizer ='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics =['accuracy']
)
model.fit(x_train_flattened,y_train,epochs=5)


# In[14]:


model.evaluate(x_test_flattened,y_test)


# In[15]:


plt.matshow(x_test[0])


# In[16]:


y_predict = model.predict(x_test_flattened)
y_predict[0]


# In[17]:


np.argmax(y_predict[0])


# In[18]:


#Confusion matricx never takes whole number check above y_predict o/p so we have to convert it into integers.
y_predict_labels = [np.argmax(i) for i in y_predict]
y_predict_labels[:5]


# In[19]:


confusion_matricx=tf.math.confusion_matrix(labels= y_test,predictions=y_predict_labels)
confusion_matricx


# In[20]:


plt.figure(figsize=(10,10))
sns.heatmap(confusion_matricx, annot=True, cmap='coolwarm',fmt='d',linewidths=1)
plt.xlabel("Predicted_Values")
plt.ylabel("Actual_Values")

