# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:58:02 2024

@author: rithv
"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\rithv\Documents\M.Tech\IITK\Summer 2023-24\Data Science\DsResearch\Banking\banking_data.csv')
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

# Cleaning the Data
print(df.head())
print(df.info())
print(df.shape)
print(df.columns)
print(df.nunique())
print(df['job'].unique())
print(df['y'].value_counts())
print(df.groupby(['default','y'])['y'].count()) # people without credit (default value no) have taken term deposit
df.drop(columns=['marital'],inplace=True)
df.info()
print(df.isnull().sum())
df.dropna(subset=['marital_status','education'],inplace=True)
print(df.isnull().sum())


# Age distribution of clients
plt.hist(df['age'],bins=30,edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Number of clients')
plt.title('Age distribution of clients')
plt.show()

# Job type variation among the clients
job_count = df['job'].value_counts()
sns.barplot(x=job_count.index,y=job_count.values,edgecolor='black')
plt.xlabel('Job')
plt.ylabel('Number of clients')
plt.title('Job type variation among the clients')
plt.xticks(rotation=90,ha='left')
plt.show()

# Marital status distribution of the clients
sns.countplot(data=df,x='marital_status',color='brown')
plt.xlabel('Marital Status')
plt.ylabel('Number of clients')
plt.title('Marital status distribution of the clients')
plt.show()

# Level of education among the clients
sns.countplot(data=df,x='education',color='blue')
plt.xlabel('Level of Education')
plt.ylabel('Number of clients')
plt.title('Level of Education of the clients')
plt.show()

# Proportion of clients having credit in default
print(df['default'].value_counts())
df.shape
credit_in_default = (815/45210)*100
print('The percentage of clients having credit in default is: ',credit_in_default)

# Average yearly balance distribution among the clients
plt.figure(figsize=(12, 6))
sns.histplot(df['balance'], bins=100, kde=True, edgecolor='black')
plt.xlabel('Average Yearly Balance')
plt.ylabel('Frequency')
plt.title('Distribution of Average Yearly Balance Among Clients')
plt.xlim(-10000, 110000)
plt.show()

# Number of clients having housing loan
df['housing'].value_counts()
print('Number of clients having housing loan: 25130')

# Number of clients having personal loan
df['loan'].value_counts()
print('Number of clients having personal loan: 7244')

# Communication types used for contacting clients during the campaign
df['contact'].unique()
print('The communication types used for contacting clients during the campaign are: Telephone, Cellular, Unknown')

# Distribution of the last contact day of the month
df['day'].nunique()
sns.histplot(df['day'],bins=31,edgecolor='black')
plt.xlabel('Last contact day of the month')
plt.ylabel('Number of clients')
plt.title('Distribution of the last contact day of the month')
plt.show()

# Variation of the last contact month of the year among the clients
df['month'].nunique()
plt.hist(df['month'],bins=12,edgecolor='black')
plt.xlabel('Last contact month')
plt.ylabel('Number of clients')
plt.title('Variation of the last contact month of the year')
plt.show()

# Distribution of the duration of the last contact
df['duration'].nunique()
sns.histplot(df['duration'], bins=50, edgecolor='black')
plt.xlabel('Duration of the last contact in seconds')
plt.ylabel('Number of clients')
plt.title('Distribution of the duration of the last contact in seconds')
plt.show()

# Contacts performed during the campaign for each client
sns.histplot(df['campaign'], edgecolor='black', bins=63)
plt.xlabel('Number of Contacts')
plt.ylabel('Number of clients')
plt.title('Contacts performed during the campaign for each client')
plt.show()

# Distribution of the number of days passed since the client was last contacted from a previous campaign
# Separate -1 values
contacted = df[df['pdays'] != -1]
not_contacted = df[df['pdays'] == -1]

plt.figure(figsize=(10, 8))
sns.histplot(contacted['pdays'], bins=50, kde=True, edgecolor='black', label='Contacted')
sns.histplot(not_contacted['pdays'], bins=1, kde=False, color='red', label='Not Contacted')
plt.xlabel('Number of Days Since Last Contact')
plt.ylabel('Number of clients')
plt.title('Distribution of Number of Days Since Last Contact')
plt.legend()
plt.show()

# Contacts performed before the current campaign for each client
sns.histplot(df['previous'], bins=50, edgecolor='black')
plt.xlabel('Contacts performed before the current campaign')
plt.ylabel('Number of clients')
plt.title('Distribution of contacts performed before the current campaign for each client')
plt.show()

# Outcomes of the previous marketing campaigns
sns.countplot(data=df,x='poutcome',color='blue')
plt.xlabel('Outcome')
plt.ylabel('Number of clients')
plt.title('Outcomes of the previous marketing campaigns')
plt.show()

# Distribution of clients who subscribed to a term deposit vs. those who did not
sns.countplot(data=df,x='y',color='blue')
plt.xlabel('Term Deposit Subscription')
plt.ylabel('Number of clients')
plt.title('Distribution of clients who subscribed to a term deposit vs. those who did not')
plt.show()

# Correlations between different attributes and the likelihood of subscribing to a term deposit
df['y'].replace({'no':0,'yes':1},inplace=True)
df['default'].replace({'no':0,'yes':1},inplace=True)
df['housing'].replace({'no':0,'yes':1},inplace=True)
df['loan'].replace({'no':0,'yes':1},inplace=True)
numeric_df = df.select_dtypes(include=['int64','float64'])
correlation_matrix = numeric_df.corr()
print(correlation_matrix)
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Banking Datatset')
plt.show()
















