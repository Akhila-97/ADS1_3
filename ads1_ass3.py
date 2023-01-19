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
                (((data['Indicator Name'] == 'Population, total') |
                   (data['Indicator Name'] == 'Total greenhouse gas '
                                'emissions (kt of CO2 equivalent)')|
                   (data['Indicator Name'] == 'CO2 emissions (metric '
                                               'tons per capita)') |
                   (data['Indicator Name'] == 'Renewable energy '
                    'consumption (% of total final energy consumption)'
                    )))]
    # return data and transposed data
    data.drop(data.columns[[-1]], axis=1, inplace=True)
    df_y = data[:]
    df_t = data.transpose()
    return df_y, df_t

def norm_func(data):
    """ Returns array normalised to [0,1].
    
        max and min data is calculate used numpy.
        Normalised data is calculated by substracting
        min value from orginal value and dividing by
        the diffrence of max and min value
        """
    max_val = np.max(data)
    min_val = np.min(data)
    norm_data = (data-min_val)/(max_val-min_val)
    return norm_data


def norm_df(data):
    """Returns all columns of the dataframe normalised.

        Calls function norm to do the normalisation  
        of one column.Columns from first to last (including)
        are normalised to [0,1]
        """
    # # iterate over all numerical columns
    for column in data.columns[1:]:
        data[column] = norm_func(data[column])
    return data


def exp_growth(time, scale_val, growth_val):
    """ Calculate exponetianl funcation
    
        Calculating exponential_function 'xf' with 
        scale value and growth value as free_parameters
        """
    
    xf = scale_val*np.exp(growth_val*(time-1960)) 
    
    return xf
def fit_plot(data, xaxis, yaxis, fit_param, xlbl, ylbl, title, cl1, cl2):
    
    """PLot the data passed in the arguments.
    
    Using pyplot the arguments that are passed
    are plotted into simple line graph"""
    
    plt.figure()
    plt.plot(data[xaxis], data[yaxis], label="Data", color=cl1)
    plt.plot(data[xaxis], data[fit_param], label="Fit", color=cl2)
    plt.legend()
    plt.title(title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.show()
    
# pass the datafile to readfile function
datafile = 'worlddata.csv'
data, data_t = readfile(datafile)
# drop unwanted columns in the dataframe
data = data.drop(['Country Code','Indicator Code'],axis =1,inplace=False)
# replace the row values for ease of use
data = data.replace(['Population, total'],'population')
data = data.replace(['Total greenhouse gas emissions (kt of CO2 equivalent)'],
                    'greenhouse gas emission')
data = data.replace(['CO2 emissions (metric tons per capita)'],'CO2 emission')
data = data.replace(['Renewable energy consumption (% of total final energy '
                     'consumption)'],'Renewable energy consumption')
data = data.reset_index()
data.drop(['index'],axis=1,inplace=True)
# find the mean value of each indicators for each country
mean_data = data.groupby(['Country Name',
                           'Indicator Name']).aggregate('mean')
# make the year as a column
df = mean_data.stack().unstack(level=1).reset_index()
# find the mean of every indicators for the time period
df1 =df.groupby(['Country Name']).aggregate('mean')
df1 = df1.reset_index()
# Get the data from 1960 to 2014
data.drop(data.iloc[:,2:36],axis=1,inplace=True)
data =data.iloc[:,0:23]
# map the correlation matrix to check the relationship
plt.figure()
heatmap = sns.heatmap(df.corr(),
                      annot=True,
                      annot_kws={'size': 16},
                      cmap='YlGnBu')
# increase the size of xticklabels and ytickslabels
heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), 
                        fontsize=16)
heatmap.set_yticklabels(heatmap.get_ymajorticklabels(),
                        fontsize=16)
plt.show()
# normalization of data
norm_data = norm_df(df1)
# Plot suplot
fig = plt.figure(figsize=(14,5))
ax = fig.add_subplot(231)    
ax1 = fig.add_subplot(232)
ax2 = fig.add_subplot(233)
# in ax draw subplot of population and co2 emission
ax.scatter(norm_data['population'],
            norm_data['CO2 emission'])
# set xlabel , ylabel and title for the plot
ax.set(xlabel='Population', ylabel='CO2 emission',
       title='co2 v population')
# in ax1 draw subplot of population and GHG
ax1.scatter(norm_data['population'],(norm_data['greenhouse gas emission']))
# set xlabel , ylabel and title for the plot
ax1.set(xlabel='Population', ylabel='greenhouse gas emission',
       title='GHG v population')
# in ax2 draw subplot of population and GHG
ax2.scatter(norm_data['population'],
            norm_data['Renewable energy consumption'],
            label = 'electric v pop')
# set xlabel , ylabel and title for the plot
ax2.set(xlabel='Population', ylabel='Renewable energy consumption',
       title='Renewable energy v population')
# arrange the data for doing clustering
kmean_data_1 = norm_data.copy()
(kmean_data_1['Renewable energy consumption']
.fillna(kmean_data_1['Renewable energy consumption'].mean(),inplace=True))
kmean_data_1.drop(['Country Name','greenhouse gas emission','CO2 emission'],
                  axis=1,inplace =True)

# calculate the silhouette score to find the number of cluster
for ic in range(2,7):
    # setup kmean and fit
    kmean = cluster.KMeans(n_clusters=ic)
    kmean.fit(kmean_data_1)
    # extract labels and calculate silhouette score
    labels = kmean.labels_
    print(ic,skmet.silhouette_score(kmean_data_1,labels))

# we get the silhouette_score of  0.6251915302652391 for 4 points
kmean = cluster.KMeans(n_clusters=4)
# Fit the data, results are stored in the kmeans object
kmean.fit(kmean_data_1)
# labels is the number of the associated clusters of (x,y) points
labels = kmean.labels_
# find the cluster centroid
cen =kmean.cluster_centers_
#plot the scatter plot using population and renewable energy
plt.figure(figsize=(7.0, 7.0))
plt.scatter(kmean_data_1["population"],
            kmean_data_1["Renewable energy consumption"], c=labels,
            cmap="Accent")
plt.title('Renewable energy consumption V population , 4 cluster',fontsize =15)
plt.xlabel('Population', fontsize=16)
plt.ylabel('Renewable energy consumption',fontsize=16)
# marking the centroid points
for ic in range(4):
        xc,yc =cen[ic,:]
        plt.plot(xc,yc,'dk',markersize=10)
        
# arrange the data for doing clustering
kmean_data_2 = norm_data.copy()
# fill the missing value with the mean
(kmean_data_2['CO2 emission'].fillna(kmean_data_2['CO2 emission'].mean(),
                                     inplace=True))
# drop unwanted columns
kmean_data_2.drop(['Country Name','greenhouse gas emission',
                   'Renewable energy consumption'],axis=1,inplace =True)
# calculate the silhouette score to find the number of cluster
for ic in range(2,7):
    # setup kmean and fit
    kmean = cluster.KMeans(n_clusters=ic)
    kmean.fit(kmean_data_2)
    # extract labels and calculate silhouette score
    labels = kmean.labels_
    print('for co2 emission',ic,skmet.silhouette_score(kmean_data_2,labels))
plt.figure()
#  we get the silhouette score of 2 with 0.7332496506370564
kmean1 = cluster.KMeans(n_clusters=2)
# Fit the data, results are stored in the kmean object
kmean1.fit(kmean_data_2)
# labels is the number of the associated clusters of (x,y) points
labels = kmean1.labels_
# find the cluster centroid
cen1 =kmean1.cluster_centers_
# plot the scatter plot for population and co2 emisison
plt.figure(figsize=(6.0, 6.0))
plt.scatter(kmean_data_2["population"], kmean_data_2["CO2 emission"],
            c=labels, cmap="Accent")
plt.title('co2 emission V population , 2 cluster',fontsize =15)
plt.xlabel('Population', fontsize=16)
plt.ylabel('co2 emission',fontsize=16)
# marking the centroid points
for ic in range(2):
    xc,yc =cen1[ic,:]
    plt.plot(xc,yc,'dk',markersize=10)

# fitting using population of USA
df_usa = df[((df['Country Name'] == 'United States') )]
df_usa=df_usa.loc[:,["level_1","population"]]
df_usa.rename({'level_1':'year'},axis=1,inplace=True)
df_usa['population'] = df_usa['population'].astype(int)
df_usa['year'] = df_usa['year'].astype(int)
df_usa=df_usa.reset_index()
df_usa.drop(['index'],axis=1,inplace=True)
# fit exponential growth with default parameters
popt, covar = opt.curve_fit(exp_growth, df_usa["year"],
                            df_usa["population"])
df_usa["pop_exp"] = exp_growth(df_usa["year"], *popt)

# Plotting the Fitting Attempt Using the values received from the 
# Curve_Fit() and default data

# Calling custom plot function defined before to plot the fit model
# Red & Yellow

fit_plot(df_usa, "year", "population", "pop_exp", "year",
         "Population, total", "Fit attempt 1", 'y', 'r')
# Finding the required start value for the time series data
# After many trials The exponential factor  with 4e8 giving better result
# Growth factor of 0.02 is giving a comprimisable fit
popt = [4e8, 0.02]
df_usa["pop_exp"] = exp_growth(df_usa["year"], *popt)
fit_plot(df_usa, "year", "population", "pop_exp", "year",
         "Population, total", "2nd Fitting with Defined Start Value", 'y', 'r')
popt, covar = opt.curve_fit(exp_growth,df_usa["year"], 
                            df_usa["population"], p0=[1.86036e+08, 0.02])
# popt is added as a new column in the dataframe
df_usa["pop_exp"] = exp_growth(df_usa["year"], *popt)
fit_plot(df_usa, "year", "population", "pop_exp", "year", 
         "Population, total", "Final Fit of Exponential Model", 'y', 'r')
sigma = np.sqrt(np.diag(covar))
# plot the function  with error range and fit
year = np.arange(1960,2020)
low, up = err.err_ranges(year, exp_growth, popt, sigma)
plt.show()
plt.plot(df_usa["year"], df_usa["population"], label="Data",color='r')
plt.plot(df_usa["year"], df_usa["pop_exp"], label="fit", color='b')
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("population,Total")
plt.title('Exponential model error range')
plt.legend()
plt.show()

# Forcasting the data using on upcoming years with upper & lower limits. 
# ie every next 10 year using exponential Model

print("Forcasted population of The United States")

low, up = err.err_ranges(2030, exp_growth, popt, sigma)
print("2030 between ", low, "and", up)

low, up = err.err_ranges(2040, exp_growth, popt, sigma)
print("2040 between ", low, "and", up)

low, up = err.err_ranges(2050, exp_growth, popt, sigma)
print("2050 between ", low, "and", up)

# inorder to forecast data until 2030
year1 = np.arange(1960, 2031)
forecast = exp_growth(year1, *popt)
low1, up1 = err.err_ranges(year1, exp_growth, popt, sigma)
plt.figure()
plt.plot(df_usa["year"], df_usa["population"], label="Data")
plt.plot(year1, forecast, label="forecast")
plt.fill_between(year1, low1, up1, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("Population, Total")
plt.title('Forecasting the data using exponential model')
plt.legend()
plt.show()