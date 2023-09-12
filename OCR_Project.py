#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[7]:


from PIL import Image
image_1 =Image.open("C:\\Users\\ACER\\Downloads\\carnumberplate.jpg")
image_1


# In[3]:


import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# In[4]:


print(image_1)


# In[5]:


x=pytesseract.image_to_string(image_1)
print(x)

