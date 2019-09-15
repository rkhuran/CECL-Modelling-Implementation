#!/usr/bin/env python
# coding: utf-8

# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.stats import boxcox
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as mp
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")

sc = StandardScaler()
df_acq = pd.read_csv('C:/Users/rohit/Documents/NCSU/FIM500/CECL project/Acquisitions_Data.csv', index_col=False)


# In[4]:


df_acq.head()


# In[5]:


df_per = pd.read_csv('C:/Users/rohit/Documents/NCSU/FIM500/CECL project/Performance_Data.csv', index_col=False)


# In[7]:


df_per.head()


# In[8]:


df_per = df_per[['LOAN_ID','Servicer.Name','LAST_RT','LAST_UPB','Loan.Age','Months.To.Legal.Mat','FCC_DTE']].reset_index(drop=True)

df_per.drop_duplicates(subset=['LOAN_ID'], keep='last', inplace=True)

df_per.head()


# In[9]:


df = pd.merge(df_acq, df_per, on='LOAN_ID')

df.drop('LOAN_ID', axis=1, inplace=True)

df.rename(index=str, columns={"FCC_DTE": 'Default'}, inplace=True)


# In[11]:


df['Default'].fillna(0, inplace=True)

df.loc[df['Default'] != 0, 'Default'] = 1

df['Default'] = df['Default'].astype(int)


# In[12]:


df_null = pd.DataFrame({'Count': df.isnull().sum(), 'Percent': 100*df.isnull().sum()/len(df)})
df_null[df_null['Count'] > 0]


# In[13]:


df.drop(['MI_PCT','CSCORE_C','MI_TYPE','Servicer.Name'], axis=1, inplace=True)

df.dropna(inplace=True)


# In[14]:


df.drop('Product.Type', axis=1, inplace=True)


# In[15]:


#Code for Data Exploration

fig, axes = mp.subplots(nrows=1, ncols=1, figsize=(8,6))

ax = df['Default'].value_counts().divide(len(df)).plot.bar(width=0.9, rot=0)
ax.set_xlabel('Default')
ax.set_ylabel('Number of Borrowers')
ax.set_ylim(0,1)


# In[16]:


fig, axes = mp.subplots(nrows=1, ncols=2, figsize=(16,6))

ax = df.groupby('Default')['ORIG_RT'].plot.hist(bins=30, density=True, alpha=0.25, ax=axes[0])
ax[0].set_ylabel('Normalized Frequency')
ax[0].set_xlabel('OrigInterestRate')
ax[0].legend()

ax = df.boxplot(by='Default', column='ORIG_RT', grid=False, widths=(0.8, 0.8), ax=axes[1])
ax.set_ylabel('OrigInterestRate')
ax.set_ylim(0,10)
ax.set_title('')
fig.suptitle('')


# In[21]:


fig, axes = mp.subplots(nrows=1, ncols=2, figsize=(16,6))

ax = df['NUM_BO'].value_counts().divide(len(df)).plot.bar(width=0.9, ax=axes[0], rot=0)
ax.set_xlabel('NumBorrow')
ax.set_ylabel('Number of Borrowers')
ax.set_ylim(0,1)

xtab = pd.pivot_table(df, index='Default', columns='NUM_BO', aggfunc='size')
xtab = xtab.div(xtab.sum(axis=1), axis=0)
ax = xtab.plot.barh(stacked=True, width=0.9, ax=axes[1])

ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
ax.set_xlabel('Fraction of Borrowers')
ax.set_ylabel('Default')
ax.set_xlim(0,1)


# In[22]:


fig, axes = mp.subplots(nrows=1, ncols=2, figsize=(16,6))

ax = df.groupby('Default')['DTI'].plot.hist(bins=30, density=True, alpha=0.25, ax=axes[0])
ax[0].set_ylabel('Normalized Frequency')
ax[0].set_xlabel('DTIRat')
ax[0].legend()

ax = df.boxplot(by='Default', column='DTI', grid=False, widths=(0.8, 0.8), ax=axes[1])
ax.set_ylabel('DTIRat')
ax.set_title('')
fig.suptitle('')


# In[23]:


fig, axes = mp.subplots(nrows=1, ncols=2, figsize=(16,6))

ax = df.groupby('Default')['CSCORE_B'].plot.hist(bins=30, density=True, alpha=0.25, ax=axes[0])
ax[0].set_ylabel('Normalized Frequency')
ax[0].set_xlabel('CreditScore')
ax[0].legend()

ax = df.boxplot(by='Default', column='CSCORE_B', grid=False, widths=(0.8, 0.8), ax=axes[1])
ax.set_ylabel('CreditScore')
ax.set_title('')
fig.suptitle('')


# In[24]:


data = df[df['ZIP_3'].isin(df['ZIP_3'].value_counts().index.tolist()[:10])]

xtab = pd.pivot_table(data, index='ZIP_3', columns='Default', aggfunc='size')
xtab = xtab.div(xtab.sum(axis=1), axis=0)
ax = xtab.plot.barh(stacked=True, width=0.9, figsize=(8,6))

ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
ax.set_xlabel('Fraction of Borrowers')
ax.set_ylabel('ZIP Code')
ax.set_xlim(0,1)


# In[25]:


fig, axes = mp.subplots(figsize=(10,8))

sns.heatmap(np.round(df.corr(), 2), annot=True, cmap='bwr',
                vmin=-1, vmax=1, square=True, linewidths=0.5)


# In[26]:


fig, axes = mp.subplots(nrows=1, ncols=2, figsize=(16,6))

df.dropna().sample(n=1000).plot.scatter(x='OLTV', y='OCLTV', ax=axes[0])
df.dropna().sample(n=1000).plot.scatter(x='LAST_UPB', y='ORIG_AMT', ax=axes[1])


# In[27]:


df.drop(['OCLTV','ORIG_AMT'], axis=1, inplace=True)

df.reset_index(drop=True, inplace=True)


# In[28]:


df['OrigDateMonth'] = df['ORIG_DTE'].apply(lambda x: x.split('/')[0].strip()).astype(object)
df['OrigDateYear'] = df['ORIG_DTE'].apply(lambda x: x.split('/')[1].strip()).astype(object)

df['FirstMonth'] = df['FRST_DTE'].apply(lambda x: x.split('/')[0].strip()).astype(object)
df['FirstYear'] = df['FRST_DTE'].apply(lambda x: x.split('/')[1].strip()).astype(object)

df.drop(['ORIG_DTE','FRST_DTE'], axis=1, inplace=True)


# In[ ]:




