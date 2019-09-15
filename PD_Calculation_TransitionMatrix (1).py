
# coding: utf-8

# In[11]:


# Group 3 # 
# Data Scope is of 2015 
# This program merges Acquisition and performance data per quarter, later drops the non-essential columns and then merges into one file
import pandas as pd
import numpy as np

acquisition = ['LoanID','Channel','SellerName','OrigInterestRate','OrigUnpPrinc','OrigLoanTerm',
               'OrigDate','FirstPayment','OrigLTV','OrigCLTV','NumBorrow','DTIRat','CreditScore',
               'FTHomeBuyer','LoanPurpose','PropertyType','NumUnits','OccStatus','PropertyState',
               'Zip','MortInsPerc','ProductType','CoCreditScore','MortInsType','RelMortInd'];
performance = ['LoanID','MonthRep','Servicer','CurrInterestRate','CurActUnpBal','LoanAge',
               'MonthsToMaturity','AdMonthsToMaturity','MaturityDate','MSA','CLDS','ModFlag',
               'ZeroBalCode','ZeroBalDate','LastInstallDate','ForeclosureDate','DispositionDate',
               'PPRC','AssetRecCost','MHRC','ATFHP','NetSaleProceeds','CreditEnhProceeds','RPMWP',
               'OFP','NIBUPB','PFUPB','RMWPF','FPWA','ServicingIndicator'];

df_acq = pd.read_csv('/Users/yuhongfu/Downloads/2015Q4/Acquisition_2015Q4.txt', sep='|', names=acquisition, index_col=False)
df_per = pd.read_csv('/Users/yuhongfu/Downloads/2015Q4/Performance_2015Q4.txt', sep='|', names=performance, index_col=False)

df_per = df_per[['LoanID','CLDS','LoanAge','CurrInterestRate','CurActUnpBal']].reset_index(drop=True)
df_acq = df_acq[['LoanID','DTIRat','CreditScore']].reset_index(drop=True)

df = pd.merge(df_acq, df_per, on='LoanID')
df.drop('LoanID', axis=1, inplace=True)
df.to_csv('/Users/yuhongfu/Downlaods/Qtr Merge/Merge_2015Q4.csv')


# In[16]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("/Users/yuhongfu/Downloads/markov_merge_2015Q4.csv")
df=df.dropna(how='any')
df=df[df.markov < '2']
df=df[df.markov > '-2']
df.drop('Unnamed: 0', axis=1, inplace=True)
df0=df[df.CLDS == '1']
df0.drop('CLDS', axis=1, inplace=True)
df0.to_csv('/Users/yuhongfu/Downloads/Markov1.csv')
X = df0.ix[:,(0,1,2,3,4)].values
y = df0.ix[:,5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix
print(classification_report(y_test, y_pred))


# In[ ]:


df = pd.read_csv("/Users/yuhongfu/Downloads/markov_merge_2015Q4.csv")
df=df.dropna(how='any')
df=df[df.markov < '2']
df=df[df.markov > '-2']
df.drop('Unnamed: 0', axis=1, inplace=True)
df0=df[df.CLDS == '1']
df0.drop('CLDS', axis=1, inplace=True)
df0.to_csv('/Users/yuhongfu/Downloads/Markov1.csv')
X = df0.ix[:,(0,1,2,3,4)].values
y = df0.ix[:,5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix
print(classification_report(y_test, y_pred))

df = pd.read_csv("/Users/yuhongfu/Downloads/markov_merge_2015Q4.csv")

df=df.dropna(how='any')
df=df[df.markov < '2']
df=df[df.markov > '-2']
df.drop('Unnamed: 0', axis=1, inplace=True)
df0=df[df.CLDS == '2']
df0.drop('CLDS', axis=1, inplace=True)
df0.to_csv('/Users/yuhongfu/Downloads/Markov2.csv')

X = df0.ix[:,(0,1,2,3,4)].values
y = df0.ix[:,5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=250)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix
print(classification_report(y_test, y_pred))


df = pd.read_csv("/Users/yuhongfu/Downloads/markov_merge_2015Q4.csv")

df=df.dropna(how='any')
df=df[df.markov < '2']
df=df[df.markov > '-2']
df.drop('Unnamed: 0', axis=1, inplace=True)
df0=df[df.CLDS == '3']
df0.drop('CLDS', axis=1, inplace=True)
df0.to_csv('/Users/yuhongfu/Downloads/Markov3.csv')

X = df0.ix[:,(0,1,2,3,4)].values
y = df0.ix[:,5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

df = pd.read_csv("/Users/yuhongfu/Downloads/markov_merge_2015Q4.csv")

df=df.dropna(how='any')
df=df[df.markov < '2']
df=df[df.markov > '-2']
df.drop('Unnamed: 0', axis=1, inplace=True)
df0=df[df.CLDS == '4']
df0.drop('CLDS', axis=1, inplace=True)
df0.to_csv('/Users/yuhongfu/Downloads/Markov4.csv')

X = df0.ix[:,(0,1,2,3,4)].values
y = df0.ix[:,5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

df = pd.read_csv("/Users/yuhongfu/Downloads/markov_merge_2015Q4.csv")

df=df.dropna(how='any')
df=df[df.markov < '2']
df=df[df.markov > '-2']
df.drop('Unnamed: 0', axis=1, inplace=True)
df0=df[df.CLDS == '5']
df0.drop('CLDS', axis=1, inplace=True)
df0.to_csv('/Users/yuhongfu/Downloads/Markov5.csv')

X = df0.ix[:,(0,1,2,3,4)].values
y = df0.ix[:,5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

df = pd.read_csv("/Users/yuhongfu/Downloads/markov_merge_2015Q4.csv")

df=df.dropna(how='any')
df=df[df.markov < '2']
df=df[df.markov > '-2']
df.drop('Unnamed: 0', axis=1, inplace=True)
df0=df[df.CLDS == '6']
df0.drop('CLDS', axis=1, inplace=True)
df0.to_csv('/Users/yuhongfu/Downloads/Markov6.csv')

X = df0.ix[:,(0,1,2,3,4)].values
y = df0.ix[:,5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

df = pd.read_csv("/Users/yuhongfu/Downloads/markov_merge_2015Q4.csv")

df=df.dropna(how='any')
df=df[df.markov < '2']
df=df[df.markov > '-2']
df.drop('Unnamed: 0', axis=1, inplace=True)
df0=df[df.CLDS == '7']
df0.drop('CLDS', axis=1, inplace=True)
df0.to_csv('/Users/yuhongfu/Downloads/Markov7.csv')

X = df0.ix[:,(0,1,2,3,4)].values
y = df0.ix[:,5].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)
y_pred = LogReg.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

