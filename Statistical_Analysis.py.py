
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as num
import scipy.stats as stats
import scipy.misc
import PyQt5


#Acquisitions_Data 
df=pd.read_csv('C:/Users/rohit/Documents/NCSU/FIM500/CECL Project/Acquisitions_Data.csv', na_values = ['no info', '.'])

#SUMMARY Statistics

# Borrower Characterstics
#-	Borrower Credit Score
#-	Co-Borrower Credit Score
#-	Debt-To-Income Ratio (DTI)
a=df[['CSCORE_B','CSCORE_C','DTI']]
print(a.describe())

#Plot of Credit Score Borrower
#h=a.CSCORE_B
#pdf = stats.norm.pdf(h, num.mean(h), num.std(h))
#plt.get_backend()
#plt.plot(h, pdf) 
#plt.show()

#Loan Characterstics 
#-	Original Interest Rate 
#-	Original Unpaid PRINCIPAL BALANCE (UPB)
#-	ORIGINAL LOAN TERM
#-	ORIGINAL LOAN-TO-VALUE (LTV) 
#-	ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)
b=df[['ORIG_RT','ORIG_AMT','ORIG_TRM','OLTV','OCLTV']]
print(b.describe())

#Performance_Data 
#Variables from Performance File 
#-	Current Interest Rate 
#-	Current Actual Unpaid Principal Balance (UPB)
#-	CURRENT LOAN DELINQUENCY STATUS
#-	LAST PAID INSTALLMENT DATE
df1=pd.read_csv('C:/Users/rohit/Documents/NCSU/FIM500/CECL Project/Performance_Data.csv',usecols=['LAST_RT','Adj.Month.To.Mat','Delq.Status'])
print(df1.describe())
