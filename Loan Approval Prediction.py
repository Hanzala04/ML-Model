#!/usr/bin/env python
# coding: utf-8

# # Loan Approval Prediction
# ## by- Hanzalah Mazhar

# In[70]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


train = pd.read_csv("C:/Users/hanzu/Downloads/train_ctrUa4K.csv")
test = pd.read_csv("C:/Users/hanzu/Downloads/test_lAUu6dG.csv")


# In[7]:


train_original = train.copy()
test_original = test.copy()


# In[8]:


train.columns


# In[9]:


test.columns


# In[10]:


train.dtypes


# In[11]:


test.dtypes


# In[12]:


train.head()


# In[13]:


train.shape


# In[14]:


test.shape


# In[15]:


train['Loan_Status'].value_counts()


# In[16]:


train['Loan_Status'].value_counts(normalize = True)


# In[17]:


train['Loan_Status'].value_counts().plot.bar()


# In[18]:


train['Gender'].value_counts()


# In[19]:


train['Gender'].value_counts(normalize = True).plot.bar()


# In[20]:


train['Married'].value_counts(normalize = True).plot.bar()


# In[21]:


train['Self_Employed'].value_counts(normalize = True).plot.bar()


# In[71]:


train['Credit_History'].value_counts(normalize = True).plot.bar(title = 'Marital Status')


# In[23]:


train['Dependents'].value_counts(normalize = True).plot.bar()


# In[24]:


train['Education'].value_counts(normalize = True).plot.bar()


# In[25]:


train['Property_Area'].value_counts(normalize = True).plot.bar()


# In[26]:


sns.distplot(train['ApplicantIncome'])


# In[27]:


train['ApplicantIncome'].plot.box()


# In[28]:


sns.distplot(train['CoapplicantIncome'])


# In[29]:


df_train = train.dropna()
sns.distplot(df_train['LoanAmount'])


# In[30]:


Gender = pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[31]:


df1 = pd.DataFrame(train, columns = ['Loan_Status', 'Gender'] ) 
df1.head()


# In[32]:


df2 = pd.crosstab(train['Gender'], train['Loan_Status'])
df2.head()
df2.div(df2.sum(1).astype(float), axis=0).plot(kind = "bar")


# In[33]:


df3 = pd.crosstab(train['Married'], train['Loan_Status'])
df3.head()
df3.div(df3.sum(1).astype(float), axis=0).plot(kind = "bar")

df4 = pd.crosstab(train['Dependents'], train['Loan_Status'])
df4.head()
df4.div(df4.sum(1).astype(float), axis=0).plot(kind = "bar")

df5 = pd.crosstab(train['Self_Employed'], train['Loan_Status'])
df5.head()
df5.div(df5.sum(1).astype(float), axis=0).plot(kind = "bar")

df6 = pd.crosstab(train['Education'], train['Loan_Status'])
df6.head()
df6.div(df6.sum(1).astype(float), axis=0).plot(kind = "bar")

df7 = pd.crosstab(train['Credit_History'], train['Loan_Status'])
df7.head()
df7.div(df7.sum(1).astype(float), axis=0).plot(kind = "bar")

df8 = pd.crosstab(train['Property_Area'], train['Loan_Status'])
df8.head()
df8.div(df8.sum(1).astype(float), axis=0).plot(kind = "bar")


# In[35]:


from sklearn.preprocessing import LabelEncoder 
labelencoder= LabelEncoder() 
train['Loan_Status'] = labelencoder.fit_transform(train['Loan_Status']) 


# In[36]:


df0 = pd.DataFrame(train, columns = ['Loan_Status', 'CoapplicantIncome', 
            'ApplicantIncome','LoanAmount',
            'Loan_Amount_Term', 'Credit_History'])
df0.head()


# In[37]:


corr = df0.corr()
corr


# In[38]:


sns.heatmap(corr)


# In[39]:


train.isnull().sum()


# In[40]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace = True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace = True)
train['Married'].fillna(train['Married'].mode()[0], inplace = True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace = True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace = True)


# In[41]:


train['Loan_Amount_Term'].value_counts()


# In[42]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace = True)


# In[43]:


train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace = True)


# In[44]:


train.isnull().sum()


# In[45]:


test.isnull().sum()


# In[46]:


test['Gender'].fillna(test['Gender'].mode()[0], inplace = True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace = True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace = True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace = True)
test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace = True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace = True)


# In[47]:


test.isnull().sum()


# In[48]:


train['LoanAmount_log'] = np.log(train['LoanAmount'])


# In[49]:


sns.distplot(train['LoanAmount_log'])


# In[50]:


sns.distplot(test['LoanAmount'])


# In[51]:


test['LoanAmount_log'] = np.log(test['LoanAmount'])
sns.distplot(test['LoanAmount_log'])


# In[52]:


train= train.drop('Loan_ID', axis=1)
test= test.drop('Loan_ID', axis=1)


# In[53]:


X = train.drop('Loan_Status', 1)
y = train['Loan_Status']


# In[54]:


X = pd.get_dummies(X)
train = pd.get_dummies(train)
test = pd.get_dummies(test)


# In[55]:


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X, y, test_size= 0.3)


# In[56]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix


# In[57]:


model = LogisticRegression()
model.fit(x_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                  intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='warn', n_jobs=None, penalty='l2',
                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)


# In[58]:


pred_cv = model.predict(x_cv)


# In[69]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_cv, pred_cv)


# In[72]:


accuracy_score(y_cv, pred_cv)


# In[63]:


pred_test = model.predict(test)


# In[64]:


submission = pd.read_csv('C:/Users/hanzu/Downloads/sample_submission_49d68Cx.csv')


# In[65]:


submission['Loan_Status'] = pred_test
submission['Loan_ID'] = test_original['Loan_ID']


# In[66]:


submission['Loan_Status'].replace(0, 'N', inplace=True)
submission['Loan_Status'].replace(1, 'Y', inplace=True)


# In[67]:


pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv('logistic.csv')

