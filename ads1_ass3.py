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
import errors as err
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
                ( ((data['Indicator Name'] == 'Population, total') |
                    (data['Indicator Name'] == 'Total greenhouse gas emissions (kt of CO2 equivalent)'
                      )|
                    (data['Indicator Name'] == 'CO2 emissions (metric tons per '
                     'capita)') |
                    (data['Indicator Name'] == 'Electric power consumption (kWh '
                     'per capita)'))
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
def exp_growth(time, scale_val, growth_val):
    """ Calculating exponential_function 'xf' with scale value and growth value as free_parameters"""
    
    xf = scale_val*np.exp(growth_val*(time-1960)) 
    
    return xf
def fit_plot(data, xaxis, yaxis, fit_param, xlbl, ylbl, title, cl1, cl2):
    
    """ Simple Fitting plot by taking data & axis values with lables and color parameters"""
    
    plt.figure()
    plt.plot(data[xaxis], data[yaxis], label="Data", color=cl1)
    plt.plot(data[xaxis], data[fit_param], label="Fit", color=cl2)

    plt.legend()
    plt.title(title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.show()
datafile = 'worlddata.csv'
data, data_t = readfile(datafile)
data = data.drop(['Country Code','Indicator Code'],axis =1,inplace=False)
data = data.reset_index()
data.drop(['index'],axis=1,inplace=True)
org_data = data.copy()
df = manipulate_df(data)
df1 =df.groupby(['Country Name']).aggregate('mean')
df1 = df1.reset_index()
data.drop(data.iloc[:,2:36],axis=1,inplace=True)
data =data.iloc[:,0:23]

# data india
# df_india = df.loc[df['Country Name'] == 'India']
# df_india.drop(['Country Name'],
#             inplace=True,
#             axis=1)
# df_us = df.loc[df['Country Name']=='United States']
# corr of india
df_corr = df.corr()
# corr of india
# df_us_corr = df_us.corr()
plt.figure()
sns.heatmap(df_corr,annot=True,cmap='YlGnBu')
plt.show()
plt.figure()
sns.heatmap(df_corr,annot=True,cmap='YlGnBu')
plt.show()
# normalization of data
norm_data = norm_df(df1)
plt.figure(figsize=(10,5))
plt.subplot(2,3,1)
plt.scatter(norm_data['Population, total'],
            norm_data['CO2 emissions (metric tons per capita)'],
            label = 'co2 v population')
plt.title('population v co2')
plt.subplot(2,3 ,2)
plt.scatter(norm_data['Population, total'],(norm_data['Total greenhouse gas '
                                                      'emissions (kt of CO2'
                                                      ' equivalent)']),
            label = 'Population v greenhouse')
plt.title('population v greenhouse')
plt.subplot(2,3 ,3)
plt.scatter(norm_data['Population, total'],
            norm_data['Electric power consumption (kWh per capita)'],
            label = 'electric v pop')
plt.title('pop v electric')

kmean_data_1 = norm_data.copy()
(kmean_data_1['Electric power consumption (kWh per capita)'].fillna(kmean_data_1['Electric power consumption (kWh per capita)'].mean(),inplace=True))
kmean_data_1.drop(['Country Name','Total greenhouse gas emissions (kt of CO2 equivalent)','CO2 emissions (metric tons per capita)'],axis=1,inplace =True)

for ic in range(2,7):
    # setup kmean and fit
    kmean = cluster.KMeans(n_clusters=ic)
    kmean.fit(kmean_data_1)
    # extract labels and calculate silhouette score
    labels = kmean.labels_
    print(ic,skmet.silhouette_score(kmean_data_1,labels))
plt.figure()
kmean = cluster.KMeans(n_clusters=3)
kmean.fit(kmean_data_1)
labels = kmean.labels_
cen =kmean.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
# Individual colours can be assigned to symbols. The label l is used to the␣
# l-th number from the colour table.
plt.scatter(kmean_data_1["Population, total"], kmean_data_1["Electric power consumption (kWh per capita)"], c=labels, cmap="Accent")
plt.title('electric power consumption')
# colour map Accent selected to increase contrast between colours
for ic in range(3):
    xc,yc =cen[ic,:]
    plt.plot(xc,yc,'dk',markersize=15)
    
    
kmean = cluster.KMeans(n_clusters=2)
kmean.fit(kmean_data_1)
labels = kmean.labels_
cen =kmean.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
    # Individual colours can be assigned to symbols. The label l is used to the␣
    # l-th number from the colour table.
plt.scatter(kmean_data_1["Population, total"], kmean_data_1["Electric power consumption (kWh per capita)"], c=labels, cmap="Accent")
plt.title('electric power consumption')
    # colour map Accent selected to increase contrast between colours
for ic in range(2):
        xc,yc =cen[ic,:]
        plt.plot(xc,yc,'dk',markersize=15)

kmean_data_2 = norm_data.copy()
(kmean_data_2['CO2 emissions (metric tons per capita)'].fillna(kmean_data_2['CO2 emissions (metric tons per capita)'].mean(),inplace=True))
kmean_data_2.drop(['Country Name','Total greenhouse gas emissions (kt of CO2 equivalent)','Electric power consumption (kWh per capita)'],axis=1,inplace =True)

for ic in range(2,7):
    # setup kmean and fit
    kmean = cluster.KMeans(n_clusters=ic)
    kmean.fit(kmean_data_2)
    # extract labels and calculate silhouette score
    labels = kmean.labels_
    print('for co2 emission',ic,skmet.silhouette_score(kmean_data_2,labels))
plt.figure()
kmean = cluster.KMeans(n_clusters=2)
kmean.fit(kmean_data_2)
labels = kmean.labels_
cen =kmean.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
# Individual colours can be assigned to symbols. The label l is used to the␣
# l-th number from the colour table.
plt.scatter(kmean_data_2["Population, total"], kmean_data_2["CO2 emissions (metric tons per capita)"], c=labels, cmap="Accent")
plt.title('co2 emission')
# colour map Accent selected to increase contrast between colours
for ic in range(2):
    xc,yc =cen[ic,:]
    plt.plot(xc,yc,'dk',markersize=15)
# data india
# df_india = df.loc[df['Country Name'] == 'India'].reset_index()
# df_india.drop(['Country Name','index','CO2 emissions (metric tons per capita)','Electric power consumption (kWh per capita)','Total greenhouse gas emissions (kt of CO2 equivalent)'],
#             inplace=True,
#             axis=1)
# df_india.rename({'level_1':'year'},axis=1,inplace=True)
# df_india['Population, total'] = pd.to_numeric(df_india['Population, total'])
# df_india["year"] = pd.to_numeric(df_india["year"])

df_india = df[((df['Country Name'] == 'India') )]
df_india=df_india.loc[:,["level_1","Population, total"]]
df_india.rename({'level_1':'year'},axis=1,inplace=True)
df_india['Population, total'] = df_india['Population, total'].astype(int)
df_india['year'] = df_india['year'].astype(int)
df_india=df_india.reset_index()
df_india.drop(['index'],axis=1,inplace=True)
# fit exponential growth with default parameters

popt, covar = opt.curve_fit(exp_growth, df_india["year"], df_india["Population, total"])
df_india["pop_exp"] = exp_growth(df_india["year"], *popt)

# Plotting the Fitting Attempt Using the values received from the Curve_Fit() and default data

print("Exponential Fit parameter", popt)

# Calling custom plot function defined before to plot the fit model Red & Yellow

fit_plot(df_india, "year", "Population, total", "pop_exp", "year",
         "Population, total", "Fit attempt 1", 'y', 'r')
# Finding the required start value for the time series data
# After many trials The exponential factor  with 4e8 giving better result
# Growth factor of 0.02 is giving a comprimisable fit

popt = [4e8, 0.02]

df_india["pop_exp"] = exp_growth(df_india["year"], *popt)
fit_plot(df_india, "year", "Population, total", "pop_exp", "year",
         "Population, total", "2nd Fitting with Defined Start Value", 'y', 'r')
popt, covar = opt.curve_fit(exp_growth,df_india["year"], 
                            df_india["Population, total"], p0=[4e8, 0.02])

df_india["pop_exp"] = exp_growth(df_india["year"], *popt)
fit_plot(df_india, "year", "Population, total", "pop_exp", "year", 
         "Population, total", "Final Fit of Exponential Model", 'y', 'r')
sigma = np.sqrt(np.diag(covar))

year = np.arange(1960,2020)
low, up = err.err_ranges(year, exp_growth, popt, sigma)
plt.show()
# fit_plot(fit_data,'Year','')
plt.plot(df_india["year"], df_india["Population, total"], label="Data",color='r')
plt.plot(df_india["year"], df_india["pop_exp"], label="fit", color='b')
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("population")
plt.legend()
plt.show()

# Forcasting the data using on upcoming years with upper & lower limits. ie every next 10 year using Logistics Model

print("Forcasted population of India")

low, up = err.err_ranges(2030, exp_growth, popt, sigma)
print("2030 between ", low, "and", up)

low, up = err.err_ranges(2040, exp_growth, popt, sigma)
print("2040 between ", low, "and", up)

low, up = err.err_ranges(2050, exp_growth, popt, sigma)
print("2050 between ", low, "and", up)
