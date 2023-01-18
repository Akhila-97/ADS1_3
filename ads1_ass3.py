# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 20:11:49 2023

@author: akhil
"""

# make the necessary imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt


def readfile(data):
    """Use to read and transopse the data.

    Transpose the dataframe after reading
    the filepath using read csv, fill the
    header with the header data from the
    transposed dataframe, and return
    both dataframes.
    """
    data = pd.read_csv(data, skiprows=4)
    data = data[
                ((data['Indicator Name'] == 'Population, total') |
                 (data['Indicator Name'] == 'Renewable energy consumption (% '
                  'of total final energy consumption)'
                  ) |
                 (data['Indicator Name'] == 'CO2 emissions (metric tons per '
                  'capita)') |
                 (data['Indicator Name'] == 'Electric power consumption (kWh '
                  'per capita)') 
                 )]
    data.drop(data.columns[[-1]], axis=1, inplace=True)
    df_y = data[:]
    df_t = data.transpose()
    return df_y, df_t


def manipulate_df(data):
    mean_data = data.groupby(['Country Name',
                              'Indicator Name']).aggregate('mean')
    df = mean_data.stack().unstack(level=1)
    df = df.reset_index()
    return df
def norm_func(data):
    max_val = np.max(data)
    min_val = np.min(data)
    norm_data = (data-min_val)/(max_val-min_val)
    return norm_data
def norm_df(data):
    for column in data.columns[1:]:
        data[column] = norm_func(data[column])
    return data
    
datafile = 'worlddata.csv'
data ,data_t = readfile(datafile)
data = data.drop(['Country Code','Indicator Code'],axis =1,inplace=False)
data = data.reset_index()
data.drop(['index'],axis=1,inplace=True)
org_data = data.copy()
data.drop(data.iloc[:,2:36],axis=1,inplace=True)
data =data.iloc[:,0:23]
df = manipulate_df(data)
df1 =df.groupby(['Country Name']).aggregate('mean')
df_orgin = df1.reset_index()
(df_orgin['Population, total'].fillna(df_orgin['Population, total'].mean(),
                                      inplace=True))
data_corr =df_orgin.corr()
print(df_orgin.isnull().any())
plt.figure()
# plt.style.use('dark_background')
sns.heatmap(data_corr,annot=True,cmap='YlGnBu')
plt.show()
# normalization of data
norm_data = norm_df(df_orgin)
plt.figure()
plt.subplot(2,3,1)
plt.scatter(norm_data['CO2 emissions (metric tons per capita)'],
            norm_data['Population, total'],label = 'co2 v population')
plt.title('co2 v population')
plt.subplot(2,3 ,2)
plt.scatter(norm_data['Population, total'],(norm_data['Renewable energy consumption (% of total final energy consumption)']),label = 'forest v population')
plt.title('renew v population')
plt.subplot(2,3 ,3)
plt.scatter(norm_data['Electric power consumption (kWh per capita)'],norm_data['Population, total'],label = 'electric v pop')
plt.title('electric v pop')