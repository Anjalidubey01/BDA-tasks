#!/usr/bin/env python
# coding: utf-8

# In[139]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')


# In[140]:


df=pd.read_csv("https://raw.githubusercontent.com/Anjalidubey01/Dataset/main/wineQualityRed_train.csv", error_bad_lines=False)


# In[141]:


bins=(2,6.99,9)
group=["bad","good"]
df['quality']=pd.cut(df['quality'],bins=bins,labels=group)


# In[142]:


label_quality=LabelEncoder()
df['quality']=label_quality.fit_transform(df['quality'])
df['quality'].value_counts()


# In[143]:


X_train=df.drop('quality',axis=1)
Y_train=df['quality']
Y_observed=Y_train
test_df=pd.read_csv("https://raw.githubusercontent.com/Anjalidubey01/Dataset/main/wineQualityRed_test.csv", error_bad_lines=False)
X_test=test_df.drop('quality',axis=1)


# In[144]:


test_df['quality']=pd.cut(test_df['quality'],bins=bins,labels=group)
test_df['quality']=label_quality.fit_transform(test_df['quality'])
Y_test=test_df['quality']
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[145]:


LOG=LogisticRegression(max_iter=1000)
LOG.fit(X_train,Y_train)
LOG_predict=LOG.predict(X_test)
Linear=LinearRegression()
Linear.fit(X_train,Y_train)
Linear_predict=Linear.predict(X_test)
SVM_classifier = SVC()
SVM_classifier.fit(X_train,Y_train)
SVM_pred = SVM_classifier.predict(X_test)
Naive_classifier=GaussianNB()
Naive_classifier.fit(X_train,Y_train)
Naive_pred = Naive_classifier.predict(X_test)


# In[146]:


mini=Linear_predict.min()-1
maxi=Linear_predict.max()+1
bins=(mini,6.999999,10)
Linear_predict=pd.cut(Linear_predict,bins,labels=group)
print("The predicted output of linear classifier for test dataset")
print(Linear_predict.value_counts())


# In[147]:


bin2=(-1,0.5,1.2)
LOG_predict=pd.cut(LOG_predict,bin2,labels=group)
SVM_pred=pd.cut(SVM_pred,bin2,labels=group)
Naive_pred=pd.cut(Naive_pred,bin2,labels=group)
print("The predicted output for test dataset by logistic classifier is:")
print(LOG_predict.value_counts())
print("The predicted output for test dataset by SVM classifier is:")
SVM_pred
print(SVM_pred.value_counts())
print("The predicted output for test dataset by Naive bayes classifier is:")
print(Naive_pred.value_counts())

