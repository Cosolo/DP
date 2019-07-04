import numpy as np 
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data=pd.read_csv('turnover.csv')

data.head()

data.describe()

data.dtypes

data.shape

### Visualization

sns.boxplot(x=data.salary, y=data.satisfaction_level)

sns.boxplot(x=data.salary, y=data.average_montly_hours)

data.columns

data.sales.unique()

data.sales.value_counts().plot.bar()

We can use label encoder to transform categorical data to numerical.

data.head()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data['sales']=le.fit_transform(data.sales)

data['salary']=le.fit_transform(data.salary)

data.head()

X=data.drop(columns='left')

y=data.left

X.head()

from sklearn.preprocessing import StandardScaler

Standardization refers to shifting the distribution of each attribute to have a mean of zero and a standard deviation of one (unit variance). It is useful to standardize attributes for a model. Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data

X_std=StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA

pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')

How many principal components do you need so that the explained variance score in total would be greater than 80%?
- I need 6 prinicipal components to have variance score bigger than 80%

sklearn_pca = PCA(n_components=6)
Y_sklearn = sklearn_pca.fit_transform(X_std)

print(Y_sklearn)



import pandas_datareader as pdr
from datetime import datetime, date

def get_stock_data (ticker, start, end):
    return pdr.get_data_yahoo(ticker,start,end)

start_date=datetime(year=2013, month=1, day=1)
end_date=datetime(year=2018, month=5, day=9)

gm=get_stock_data('GM', start_date, end_date)

gm.head()

gm.head()

gm.dtypes

gm_adjClose=gm['Adj Close']

gm_adjClose=pd.DataFrame(gm['Adj Close'])

gm_adjClose.head()

# first plot it
gm_adjClose['Adj Close'].plot()

# pretp da je cijena stocka close.

data_weekly=gm['Adj Close'].resample('W-MON').mean()

data_weekly=pd.DataFrame(data_weekly)

data_weekly

plt.figure(figsize=(15,10))
plt.plot(gm_adjClose['Adj Close'])
plt.xlabel('Time period')
plt.ylabel('Value')
plt.title('General Motors price weekly on Monday')



gm['Daily changes']=gm['Adj Close']/gm['Adj Close'].shift(1)-1

gm.head()

import math

apple['Daily changes']=gm['Daily changes'].fillna(0)
def func(gm):
    if gm['Daily changes'] == 0:
        return 'SAME'
    elif gm['Daily changes'] > 0:
        return 'UP'
    else:
        return 'DOWN'

gm['Daily RETURNS']=gm.apply(func, axis=1)

gm.head()

gm['Daily RETURNS'].value_counts().plot.bar()

gm_monthly=gm['Adj Close'].asfreq('M').ffill()

gm_monthly=pd.DataFrame(gm_monthly)

gm_monthly.head()

Calculate the simple monthly percentage changes and compare that number to the proportion of “UP” movements you found in the previous question. 

gm_monthly['Monthly changes']=gm_monthly/gm_monthly.shift(1)-1

gm_monthly.head()

gm_monthly['Monthly changes']=gm_monthly['Monthly changes'].fillna(0)

def func(gm_monthly):
    if gm_monthly['Monthly changes'] == 0:
        return 'SAME'
    elif gm_monthly['Monthly changes'] > 0:
        return 'UP'
    else:
        return 'DOWN'

gm_monthly.head()

gm_monthly['Monthly RETURN']=apple_monthly.apply(func, axis=1)

gm_monthly.head()

gm_monthly['Monthly RETURN'].value_counts().plot.bar()

gm['Daily RETURNS'].value_counts()

gm_monthly['Monthly RETURN'].value_counts()

print('Percentage UP on daily : ', 699/(699+645+4)*100,'%')
print('Percentage UP on monthly : ', 31/(31+20+13)*100,'%')
