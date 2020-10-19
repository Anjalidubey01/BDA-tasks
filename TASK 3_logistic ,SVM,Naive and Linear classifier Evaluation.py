#!/usr/bin/env python
# coding: utf-8

# In[43]:


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


# In[44]:


df=pd.read_csv("https://raw.githubusercontent.com/Anjalidubey01/Dataset/main/wineQualityRed_train.csv", error_bad_lines=False)


# In[45]:


bins=(2,6.99,9)
group=["bad","good"]
df['quality']=pd.cut(df['quality'],bins=bins,labels=group)


# In[46]:


label_quality=LabelEncoder()
df['quality']=label_quality.fit_transform(df['quality'])
df['quality'].value_counts()


# In[47]:


X_train=df.drop('quality',axis=1)
Y_train=df['quality']
Y_observed=Y_train
test_df=pd.read_csv("https://raw.githubusercontent.com/Anjalidubey01/Dataset/main/wineQualityRed_test.csv", error_bad_lines=False)
X_test=test_df.drop('quality',axis=1)


# In[48]:


test_df['quality']=pd.cut(test_df['quality'],bins=bins,labels=group)
test_df['quality']=label_quality.fit_transform(test_df['quality'])
Y_test=test_df['quality']
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[49]:


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


# In[50]:


mini=Linear_predict.min()-1
maxi=Linear_predict.max()+1
bins=(mini,6.999999,10)
Linear_predict=pd.cut(Linear_predict,bins,labels=group)

# In[51]:


labelencoder_y = LabelEncoder()
Linear_predict= labelencoder_y.fit_transform(Linear_predict)
from sklearn.metrics import confusion_matrix
print("The heatmap of confusion matrix for linear classifier is :")
cm = confusion_matrix(Y_test, Linear_predict)
sns.heatmap(cm,annot=True,fmt='2.0f')


# In[52]:


print("The heatmap of confusion matrix for logistic classifier is :")
cm1 = confusion_matrix(Y_test, LOG_predict)
sns.heatmap(cm1,annot=True,fmt='2.0f')


# In[53]:


print("The heatmap of confusion matrix for Naive bayesian classifier is :")
cm2 = confusion_matrix(Y_test, Naive_pred)
sns.heatmap(cm2,annot=True,fmt='2.0f')


# In[54]:


print("The heatmap of confusion matrix for SVM classifier is :")
cm3= confusion_matrix(Y_test,SVM_pred)
sns.heatmap(cm3,annot=True,fmt='2.0f')


# In[58]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
Linear_accuracy=(cm[0,0]+cm[1,1])/(sum(sum(cm)))
Linear_F1=f1_score(Y_test, Linear_predict, average="macro")
Linear_Precision=precision_score(Y_test,Linear_predict, average="macro")
Linear_recall=recall_score(Y_test,Linear_predict, average="macro")
Linear_sensitivity=cm[0,0]/(cm[0,0]+cm[0,1])
Linear_specificity=cm[1,1]/(cm[1,0]+cm[1,1])
print("For linear classifier:")
print("Accuracy ", Linear_accuracy)
print("F1 SCORE ",Linear_F1)
print("Precision",Linear_Precision)
print("Recall ",Linear_recall)
print("Sensitivity ",Linear_sensitivity)
print("Specificity ",Linear_specificity)
print("\n")
LOG_accuracy=(cm1[0,0]+cm1[1,1])/(sum(sum(cm1)))
LOG_F1=f1_score(Y_test, LOG_predict, average="macro")
LOG_Precision=precision_score(Y_test,LOG_predict, average="macro")
LOG_recall=recall_score(Y_test,LOG_predict, average="macro")
LOG_sensitivity=cm1[0,0]/(cm1[0,0]+cm1[0,1])
LOG_specificity=cm1[1,1]/(cm1[1,0]+cm1[1,1])
print("For logistic classifier:")
print("Accuracy ", LOG_accuracy)
print("F1 SCORE ",LOG_F1)
print("Precision",LOG_Precision)
print("Recall ",LOG_recall)
print("Sensitivity ",LOG_sensitivity)
print("Specificity ",LOG_specificity)
print("\n")
SVM_accuracy=(cm3[0,0]+cm3[1,1])/(sum(sum(cm3)))
SVM_F1=f1_score(Y_test, SVM_pred, average="macro")
SVM_Precision=precision_score(Y_test,SVM_pred, average="macro")
SVM_recall=recall_score(Y_test,SVM_pred, average="macro")
SVM_sensitivity=cm3[0,0]/(cm3[0,0]+cm3[0,1])
SVM_specificity=cm3[1,1]/(cm3[1,0]+cm3[1,1])
print("For SVM classifier:")
print("Accuracy ", SVM_accuracy)
print("F1 SCORE ",SVM_F1)
print("Precision",SVM_Precision)
print("Recall ",SVM_recall)
print("Sensitivity ",SVM_sensitivity)
print("Specificity ",SVM_specificity)
print("\n")
Naive_accuracy=(cm2[0,0]+cm2[1,1])/(sum(sum(cm2)))
Naive_F1=f1_score(Y_test, Naive_pred, average="macro")
Naive_Precision=precision_score(Y_test,Naive_pred, average="macro")
Naive_recall=recall_score(Y_test,Naive_pred, average="macro")
Naive_sensitivity=cm2[0,0]/(cm2[0,0]+cm2[0,1])
Naive_specificity=cm2[1,1]/(cm2[1,0]+cm2[1,1])
print("For Naive bayesian classifier:")
print("Accuracy ", Naive_accuracy)
print("F1 SCORE ",Naive_F1)
print("Precision",Naive_Precision)
print("Recall ",Naive_recall)
print("Sensitivity ",Naive_sensitivity)
print("Specificity ",Naive_specificity)
print("\n")


# In[ ]:




