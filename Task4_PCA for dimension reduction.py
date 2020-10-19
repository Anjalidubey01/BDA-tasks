#!/usr/bin/env python
# coding: utf-8

# In[138]:


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report


# In[139]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
df=pd.read_csv("https://raw.githubusercontent.com/Anjalidubey01/Dataset/main/wineQualityRed_train.csv", error_bad_lines=False)
X_train=df.drop('quality',axis=1)


# In[140]:


test_df=pd.read_csv("https://raw.githubusercontent.com/Anjalidubey01/Dataset/main/wineQualityRed_test.csv", error_bad_lines=False)
X_test=test_df.drop('quality',axis=1)


# In[141]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[142]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 7)
redwine_7_training= pca.fit_transform(X_train)
redwine_7_testing = pca.transform(X_test)


# In[143]:


import seaborn as sns
correlation=df.corr()
print("The correlation matrix is ")
print(correlation)
sns.heatmap(correlation)
print("\n")

# In[144]:


pca2 = PCA(n_components = 4)
redwine_4_training= pca.fit_transform(X_train)
redwine_4_testing = pca.transform(X_test)


# In[145]:


bins=(2,6.99,9)
group=["bad","good"]


# In[146]:


df['quality']=pd.cut(df['quality'],bins=bins,labels=group)


# In[147]:


test_df['quality']=pd.cut(test_df['quality'],bins=bins,labels=group)
Y_train=df['quality']
Y_test=test_df['quality']
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
Y_train= labelencoder_y.fit_transform(Y_train)
Y_test=labelencoder_y.fit_transform(Y_test)


# In[148]:


SVM_7=SVC()
SVM_7.fit(redwine_7_training,Y_train)
SVM_7_PRED=SVM_7.predict(redwine_7_testing)


# In[149]:


from sklearn.metrics import classification_report
SVM7_report=classification_report(Y_test,SVM_7_PRED,output_dict=True)


# In[150]:


SVM7_report=pd.DataFrame(SVM7_report).transpose()
print("Classification report of SVM classifier with 7 attributes:")
print(SVM7_report)
print("\n")


# In[151]:


SVM_4=SVC()
SVM_4.fit(redwine_4_training,Y_train)
SVM_4_PRED=SVM_4.predict(redwine_4_testing)
SVM4_report=classification_report(Y_test,SVM_4_PRED,output_dict=True)
SVM4_report=pd.DataFrame(SVM4_report).transpose()
print("Classification report of SVM classifier with 4 attributes:")
print(SVM4_report)
print("\n")

# In[152]:


Naive_7=GaussianNB()
Naive_7.fit(redwine_7_training,Y_train)
Naive_7_PRED=Naive_7.predict(redwine_7_testing)
Naive7_report=classification_report(Y_test,Naive_7_PRED,output_dict=True)
Naive7_report=pd.DataFrame(Naive7_report).transpose()
print("Classification report of Naive Byesian classifier with 7 attributes:")
print(Naive7_report)
print("\n")

# In[153]:


Naive_4=GaussianNB()
Naive_4.fit(redwine_4_training,Y_train)
Naive_4_PRED=Naive_4.predict(redwine_4_testing)
Naive4_report=classification_report(Y_test,Naive_4_PRED,output_dict=True)
Naive4_report=pd.DataFrame(Naive4_report)
print("Classification report of Naive Byesian classifier with 4 attributes:")
print(Naive4_report)
print("\n")

# In[154]:


LOG_7=LogisticRegression(max_iter=1000)
LOG_7.fit(redwine_7_training,Y_train)
LOG_7_PRED=LOG_7.predict(redwine_7_testing)
LOG7_report=classification_report(Y_test,LOG_7_PRED,output_dict=True)
LOG7_report=pd.DataFrame(LOG7_report).transpose()
print("Classification report of logistic classifier with 7 attributes:")
print(LOG7_report)
print("\n")

# In[155]:


LOG_4=LogisticRegression(max_iter=1000)
LOG_4.fit(redwine_4_training,Y_train)
LOG_4_PRED=LOG_7.predict(redwine_4_testing)
LOG4_report=classification_report(Y_test,LOG_4_PRED,output_dict=True)
LOG4_report=pd.DataFrame(LOG4_report).transpose()
print("Classification report of logistic classifier with 4 attributes:")
print(LOG4_report)
print("\n")

# In[156]:


import warnings
warnings.filterwarnings('ignore')


# In[157]:


Linear7=LinearRegression()
Linear7.fit(redwine_7_training,Y_train)
Linear_7_PRED=Linear7.predict(redwine_7_testing)


# In[158]:


mini=Linear_7_PRED.min()-1
maxi=Linear_7_PRED.max()+1


# In[159]:


bins=(mini,6.999999,10)
Linear_7_PRED=pd.cut(Linear_7_PRED,bins,labels=group)
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
Linear_7_PRED = labelencoder_y.fit_transform(Linear_7_PRED)


# In[160]:


from sklearn.metrics import classification_report
Linear7_report=classification_report(Y_test,Linear_7_PRED,output_dict=True)
Linear7_report=pd.DataFrame(Linear7_report).transpose()
print("Classification report of linear classifier with 7 attributes:")
print(Linear7_report)
print("\n")

# In[161]:


Linear4=LinearRegression()
Linear4.fit(redwine_4_training,Y_train)
Linear_4_PRED=Linear4.predict(redwine_4_testing)


# In[162]:


mini1=Linear_4_PRED.min()-1
maxi1=Linear_4_PRED.max()+1


# In[163]:


bins=(mini1,6.999999,10)
Linear_4_PRED=pd.cut(Linear_4_PRED,bins,labels=group)
labelencoder_y = LabelEncoder()
Linear_4_PRED = labelencoder_y.fit_transform(Linear_4_PRED)


# In[164]:


Linear4_report=classification_report(Y_test,Linear_4_PRED,output_dict=True)
Linear4_report=pd.DataFrame(Linear4_report).transpose()
print("Classification report of linear regression classifier with 4 attributes:")
print(Linear4_report)
print("\n")

# In[165]:


Linear7_cm = confusion_matrix(Y_test, Linear_7_PRED)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
Linear7_accuracy=(Linear7_cm[0,0]+Linear7_cm[1,1])/(sum(sum(Linear7_cm)))
Linear7_F1=f1_score(Y_test,Linear_7_PRED, average="macro")
Linear7_Precision=precision_score(Y_test,Linear_7_PRED, average="macro")
Linear7_recall=recall_score(Y_test,Linear_7_PRED, average="macro")
Linear7_sensitivity=Linear7_cm[0,0]/(Linear7_cm[0,0]+Linear7_cm[0,1])
Linear7_specificity=Linear7_cm[1,1]/(Linear7_cm[1,0]+Linear7_cm[1,1])
print("For linear classifier with 7 attributes:")
print("Accuracy ", Linear7_accuracy)
print("F1 SCORE ",Linear7_F1)
print("Precision",Linear7_Precision)
print("Recall ",Linear7_recall)
print("Sensitivity ",Linear7_sensitivity)
print("Specificity ",Linear7_specificity)
print("\n")

# In[166]:


Linear4_cm = confusion_matrix(Y_test, Linear_4_PRED)
Linear4_accuracy=(Linear4_cm[0,0]+Linear4_cm[1,1])/(sum(sum(Linear4_cm)))
Linear4_F1=f1_score(Y_test,Linear_4_PRED, average="macro")
Linear4_Precision=precision_score(Y_test,Linear_4_PRED, average="macro")
Linear4_recall=recall_score(Y_test,Linear_4_PRED, average="macro")
Linear4_sensitivity=Linear4_cm[0,0]/(Linear4_cm[0,0]+Linear4_cm[0,1])
Linear4_specificity=Linear4_cm[1,1]/(Linear4_cm[1,0]+Linear4_cm[1,1])
print("For linear classifier with 4 attributes:")
print("Accuracy ", Linear4_accuracy)
print("F1 SCORE ",Linear4_F1)
print("Precision",Linear4_Precision)
print("Recall ",Linear4_recall)
print("Sensitivity ",Linear4_sensitivity)
print("Specificity ",Linear4_specificity)
print("\n")

# In[167]:


LOG7_cm = confusion_matrix(Y_test, LOG_7_PRED)
LOG7_accuracy=(LOG7_cm[0,0]+LOG7_cm[1,1])/(sum(sum(LOG7_cm)))
LOG7_F1=f1_score(Y_test,LOG_7_PRED, average="macro")
LOG7_Precision=precision_score(Y_test,LOG_7_PRED, average="macro")
LOG7_recall=recall_score(Y_test,LOG_7_PRED, average="macro")
LOG7_sensitivity=LOG7_cm[0,0]/(LOG7_cm[0,0]+LOG7_cm[0,1])
LOG7_specificity=LOG7_cm[1,1]/(LOG7_cm[1,0]+LOG7_cm[1,1])
print("For logistic classifier with 7 attributes:")
print("Accuracy ", LOG7_accuracy)
print("F1 SCORE ",LOG7_F1)
print("Precision",LOG7_Precision)
print("Recall ",LOG7_recall)
print("Sensitivity ",LOG7_sensitivity)
print("Specificity ",LOG7_specificity)
print("\n")

# In[168]:


LOG4_cm = confusion_matrix(Y_test, LOG_4_PRED)
LOG4_accuracy=(LOG4_cm[0,0]+LOG4_cm[1,1])/(sum(sum(LOG4_cm)))
LOG4_F1=f1_score(Y_test,LOG_4_PRED, average="macro")
LOG4_Precision=precision_score(Y_test,LOG_4_PRED, average="macro")
LOG4_recall=recall_score(Y_test,LOG_4_PRED, average="macro")
LOG4_sensitivity=LOG4_cm[0,0]/(LOG4_cm[0,0]+LOG4_cm[0,1])
LOG4_specificity=LOG4_cm[1,1]/(LOG4_cm[1,0]+LOG4_cm[1,1])
print("For logistic classifier with 4 attributes:")
print("Accuracy ", LOG4_accuracy)
print("F1 SCORE ",LOG4_F1)
print("Precision",LOG4_Precision)
print("Recall ",LOG4_recall)
print("Sensitivity ",LOG4_sensitivity)
print("Specificity ",LOG4_specificity)
print("\n")

# In[169]:


SVM7_cm = confusion_matrix(Y_test, SVM_7_PRED)
SVM7_accuracy=(SVM7_cm[0,0]+SVM7_cm[1,1])/(sum(sum(SVM7_cm)))
SVM7_F1=f1_score(Y_test,SVM_7_PRED, average="macro")
SVM7_Precision=precision_score(Y_test,SVM_7_PRED, average="macro")
SVM7_recall=recall_score(Y_test,SVM_7_PRED, average="macro")
SVM7_sensitivity=SVM7_cm[0,0]/(SVM7_cm[0,0]+SVM7_cm[0,1])
SVM7_specificity=SVM7_cm[1,1]/(SVM7_cm[1,0]+SVM7_cm[1,1])
print("For SVM classifier with 7 attributes:")
print("Accuracy ", SVM7_accuracy)
print("F1 SCORE ",SVM7_F1)
print("Precision",SVM7_Precision)
print("Recall ",SVM7_recall)
print("Sensitivity ",SVM7_sensitivity)
print("Specificity ",SVM7_specificity)
print("\n")

# In[170]:


SVM4_cm = confusion_matrix(Y_test, SVM_7_PRED)
SVM4_accuracy=(SVM4_cm[0,0]+SVM4_cm[1,1])/(sum(sum(SVM4_cm)))
SVM4_F1=f1_score(Y_test,SVM_4_PRED, average="macro")
SVM4_Precision=precision_score(Y_test,SVM_4_PRED, average="macro")
SVM4_recall=recall_score(Y_test,SVM_4_PRED, average="macro")
SVM4_sensitivity=SVM4_cm[0,0]/(SVM4_cm[0,0]+SVM4_cm[0,1])
SVM4_specificity=SVM4_cm[1,1]/(SVM4_cm[1,0]+SVM4_cm[1,1])
print("For SVM classifier with 4 attributes:")
print("Accuracy ", SVM4_accuracy)
print("F1 SCORE ",SVM4_F1)
print("Precision",SVM4_Precision)
print("Recall ",SVM4_recall)
print("Sensitivity ",SVM4_sensitivity)
print("Specificity ",SVM4_specificity)
print("\n")

# In[171]:


Naive7_cm = confusion_matrix(Y_test,Naive_7_PRED)
Naive7_accuracy=(Naive7_cm[0,0]+Naive7_cm[1,1])/(sum(sum(Naive7_cm)))
Naive7_F1=f1_score(Y_test,Naive_7_PRED, average="macro")
Naive7_Precision=precision_score(Y_test,Naive_7_PRED, average="macro")
Naive7_recall=recall_score(Y_test,Naive_7_PRED, average="macro")
Naive7_sensitivity=Naive7_cm[0,0]/(Naive7_cm[0,0]+Naive7_cm[0,1])
Naive7_specificity=Naive7_cm[1,1]/(Naive7_cm[1,0]+Naive7_cm[1,1])
print("For Naive Bayes classifier with 7 attributes:")
print("Accuracy ", Naive7_accuracy)
print("F1 SCORE ",Naive7_F1)
print("Precision",Naive7_Precision)
print("Recall ",Naive7_recall)
print("Sensitivity ",Naive7_sensitivity)
print("Specificity ",Naive7_specificity)
print("\n")

# In[172]:


Naive4_cm = confusion_matrix(Y_test,Naive_4_PRED)
Naive4_accuracy=(Naive4_cm[0,0]+Naive4_cm[1,1])/(sum(sum(Naive4_cm)))
Naive4_F1=f1_score(Y_test,Naive_4_PRED, average="macro")
Naive4_Precision=precision_score(Y_test,Naive_4_PRED, average="macro")
Naive4_recall=recall_score(Y_test,Naive_4_PRED, average="macro")
Naive4_sensitivity=Naive4_cm[0,0]/(Naive4_cm[0,0]+Naive4_cm[0,1])
Naive4_specificity=Naive4_cm[1,1]/(Naive4_cm[1,0]+Naive4_cm[1,1])
print("For Naive Bayes classifier with 4 attributes:")
print("Accuracy ", Naive4_accuracy)
print("F1 SCORE ",Naive4_F1)
print("Precision",Naive4_Precision)
print("Recall ",Naive4_recall)
print("Sensitivity ",Naive4_sensitivity)
print("Specificity ",Naive4_specificity)
print("\n")

# In[ ]:




