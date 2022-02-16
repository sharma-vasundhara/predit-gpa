#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression


# # Simple Linear Regression

# In[3]:


data = pd.read_csv('../input/101-simple-linear-regressioncsv/1.01. Simple linear regression.csv')
data.head()


# In[4]:


# feature
x = data['SAT']

# target
y = data['GPA']


# In[5]:


x_matrix = x.values.reshape(-1, 1)
reg = LinearRegression()
reg.fit(x_matrix, y)


# In[6]:


# R-squared
display(reg.score(x_matrix, y))

# coefficiants
display(reg.coef_)

# intercept
display(reg.intercept_)


# In[7]:


new_data = pd.DataFrame(data = [1730, 1750], columns = ['SAT'])
reg.predict(new_data)


# In[8]:


new_data['Predicated_GPA'] = reg.predict(new_data)
new_data


# In[9]:


plt.scatter(x, y)
yhat = reg.coef_ * x_matrix + reg.intercept_

fig = plt.plot(x, yhat, lw = 4, c = 'orange', label = 'Regression Line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()


# # Multiple Linear Regression

# In[10]:


data = pd.read_csv('../input/102-multiple-linear-regression/1.02 Multiple linear regression.csv')
data.head()


# In[11]:


data.describe()


# In[12]:


x = data[['SAT', 'Rand 1,2,3']]
y = data['GPA']


# In[13]:


reg.fit(x, y)

# R-squared
r2 = reg.score(x, y)
display(reg.score(x, y))

# coefficiants
display(reg.coef_)

# intercept
display(reg.intercept_)


# In[14]:


n = x.shape[0]
p = x.shape[1]

adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
adjusted_r2


# In[15]:


from sklearn.feature_selection import f_regression


# In[16]:


p_values = f_regression(x, y)[1]
p_values.round(3)


# In[17]:


reg_summary = pd.DataFrame(data = x.columns.values, columns = ['Features'])
reg_summary['Coefficiants'] = reg.coef_
reg_summary['P-values'] = p_values.round(3)

reg_summary


# In[18]:


plt.scatter(x['SAT'], y)
yhat = reg.coef_ * x + reg.intercept_

fig = plt.plot(x, yhat, lw = 4, c = 'orange', label = 'Regression Line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.ylim(2.25,4)
plt.xlim(1600, 2100)
plt.show()


# # Multiple Linear Regression with standardization

# In[19]:


from sklearn.preprocessing import StandardScaler


# In[20]:


scaler = StandardScaler()
scaler.fit(x)


# In[21]:


x_scaled = scaler.transform(x)

reg = LinearRegression()
reg.fit(x_scaled, y)


# In[22]:


reg.coef_, reg.intercept_


# In[23]:


reg_summary = pd.DataFrame([['Bias'], ['SAT'], ['Rand 1,2,3']], columns = ['Features'])
reg_summary['Weights'] = reg.intercept_, reg.coef_[0], reg.coef_[1]

reg_summary


# ### Making predictions with standardized coefficiants

# In[24]:


new_data = pd.DataFrame([[1700, 2], [1750, 3]], columns = ['SAT', 'Rand 1,2,3'])
new_data


# In[25]:


new_scaled_data = scaler.transform(new_data)
reg.predict(new_scaled_data)


# In[26]:


reg_simple = LinearRegression()
x_simple_matrix = x_scaled[:,0].reshape(-1, 1)
reg_simple.fit(x_simple_matrix, y)


# In[27]:


reg_simple.predict(new_scaled_data[:,0].reshape(-1, 1))

