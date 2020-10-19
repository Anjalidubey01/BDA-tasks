#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[67]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[68]:


df=pd.read_csv("https://raw.githubusercontent.com/Anjalidubey01/Dataset/main/wineQualityRed_train.csv", error_bad_lines=False)


# In[69]:


sns.heatmap(df.corr())
print("Correlations of attributes with quality")
plt.show()


# In[70]:


y_observed=df['quality']
x_observed=df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]


# In[71]:


regressor = LinearRegression()
regressor.fit(x_observed,y_observed)
features=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']


# In[72]:


coeff=pd.DataFrame(regressor.coef_,features)


# In[73]:


print('Considering all the attributes the coefficients of the linear regression model are:')
coeff.columns = ['Coeffecient']
print(coeff)


# In[74]:


print('The intercept of this linear model is')
print(regressor.intercept_)


# In[75]:


test_df=pd.read_csv("https://raw.githubusercontent.com/Anjalidubey01/Dataset/main/wineQualityRed_test.csv", error_bad_lines=False)
y_test=test_df['quality']
x_test=test_df[features]


# In[76]:


train_pred=regressor.predict(x_observed)
test_pred=regressor.predict(x_test)


# In[77]:


from sklearn.metrics import mean_squared_error
mse_test=mean_squared_error(test_pred,y_test)
print('Mean squared error for the test dataset is ',mse_test)
print('sum of squared error for the dataset is',mse_test*1119)#number of tuples are 1119


# In[78]:


#Now doing linear regression with respect to only one attribute that is pH
reg = LinearRegression()
reg.fit(df[['pH']],df.quality)
quality_pred=reg.predict(df[['pH']])
print('The coefficient for linear regression model with respect to only one attribute that is pH',reg.coef_)


# In[79]:


print('The intercept for this model is',reg.intercept_)


# In[80]:


print('Line fitting over the training dataset')
plt.xlabel('pH',fontsize=20)
plt.ylabel('quality',fontsize=20)
plt.scatter(df.pH,df.quality,color='red')
plt.plot(df.pH,quality_pred,color='blue')


# In[81]:


test_pred1=reg.predict(test_df[['pH']])


# In[82]:


mse_test1=mean_squared_error(test_pred1,test_df['quality'])
print('Mean squared error for the test dataset is ',mse_test1)
print('sum of squared error for the dataset is',mse_test1*1119)#number of tuples are 1119


# In[ ]:




