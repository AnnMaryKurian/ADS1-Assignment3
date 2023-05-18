# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:07:56 2023

@author: mypc
"""

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.datasets as skdat
import cluster_tools as ct
import errors as err
import numpy as np
import scipy.optimize as opt


def read_data(filename):
    '''
    Reading and performing file manipulation is done here.
    The file passed from the function call is received in variable named filename.
    '''
    dframe = pd.read_csv(filename, skiprows=4)
    return dframe


def datafilter(dframe):
    '''
    The arguments passed are received in parameter called dframe
    Indexes are reset for new dataframe.
    data for specific country is fetched.
    '''
    df_pop = dframe.loc[:, 'Country Name':'2020']
    df_pop = df_pop.drop(['Country Code', 'Indicator Name',
                         'Indicator Code', '1960'], axis=1)
    df_pop_T = df_pop.transpose()
    df_pop_T = df_pop_T.rename({'Country Name': 'Years'})
    df_pop_T = df_pop_T.reset_index(drop=False)
    df_pop_T.columns = df_pop_T.iloc[0, :]
    df_pop_T = df_pop_T.drop(0)
    # converting data type
    df_pop_T['Australia'] = df_pop_T['Australia'].astype(float)
    df_pop_T['Years'] = df_pop_T['Years'].astype(float)
    df_pop_T.to_csv("data123.csv")
    return df_pop_T

    # function for clustering of data


def callingcluster(datafr, strtil, val1, val2, clstitle, x, y, clusterfig):
    # plotting the scatter matrix
    pd.plotting.scatter_matrix(datafr, figsize=(12, 12), s=5, alpha=0.8)
    plt.savefig(strtil + '.png')
    plt.show()

    df_ex = datafr[[val1, val2]]
    df_ex = df_ex.dropna()
    df_ex = df_ex.reset_index()
    df_ex = df_ex.drop("index", axis=1)
    # finding the minimum and maximum values
    df_norm, df_min, df_max = ct.scaler(df_ex)

# looping over the number of clusters
    for ncluster in range(2, 10):

        kmeans = cluster.KMeans(n_clusters=ncluster)

# data fitting and storing in kmeans object
    kmeans.fit(df_norm)
    labels = kmeans.labels_
# extract the  cluster centres
    cen = kmeans.cluster_centers_
    ncluster = 5
# set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_norm)
    labels = kmeans.labels_
# extract the cluster centres
    cen = kmeans.cluster_centers_
    cen = np.array(cen)
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('gist_rainbow')
# plotting the scatter plot using val1 and val2 arguments
    plt.scatter(df_norm[val1], df_norm[val2], 30, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 100, "k", label='centroid', marker="*")
# setting the title and x and y labels
    plt.title(clstitle, fontsize=20)
    plt.xlabel(x, fontsize=15)
    plt.ylabel(y, fontsize=15)
    plt.legend(fontsize=20, loc='upper left')
    plt.show()

# backscale function to convert the cluster centre
    scen = ct.backscale(cen, df_min, df_max)
    xcen = scen[:, 0]
    ycen = scen[:, 1]
# plotting new cluster after backscale
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('gist_rainbow')
    plt.scatter(df_ex[val1], df_ex[val2], 30, labels, marker="o", cmap=cm)
# placing the centroids in position
    plt.scatter(xcen, ycen, 100, "k", label='centroid', marker="*")
    plt.title(clstitle, fontsize=20)
    plt.xlabel(x, fontsize=18)
    plt.ylabel(y, fontsize=18)
    plt.legend(fontsize=20, loc='upper left')
    plt.savefig(clusterfig + '.png')
    plt.show()
    return


def logistic(t, n0, g, t0):

    # Calculates the logistic function
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


def fitting(df_crl):
    '''
    The arguments passed are received in parameter called df_crl.
    obtaining the curve fit using function.
    '''
    param, covar = opt.curve_fit(
        logistic, df_crl["Years"], df_crl["Australia"], p0=(1.2e12, 0.03, 1961.0))
    sigma = np.sqrt(np.diag(covar))
    df_crl["fit"] = logistic(df_crl["Years"], *param)
# plotting the fitted graph
    df_crl.plot("Years", ["Australia", "fit"])
    plt.title("Fitting for Cereal land in Australia", fontsize=20)
    plt.xlabel("years")
    plt.ylabel("Cereal Land")
    plt.savefig("logfitting" + '.png')
    plt.show()

    param, covar = opt.curve_fit(
        logistic, df_crl["Years"], df_crl["Australia"], p0=(1.2e12, 0.03, 1961.0))
    sigma = np.sqrt(np.diag(covar))
    year = np.arange(1960, 2040)
# logistic function is used and forecasting is done for year 2040
    forecast = logistic(year, *param)
    plt.figure()
# plotting the fitted image showing forecasting
    plt.plot(df_crl["Years"], df_crl["Australia"], label="Australia")
    plt.plot(year, forecast, label="Forecast", color="brown")
    plt.title("Forecast for Australia", fontsize=20)
    plt.xlabel("years")
    plt.ylabel("Cereal Land")
    plt.legend(loc='upper left')
    plt.savefig("forecast" + '.png')
    plt.show()


# finding the error ranges for the country
    low, up = err.err_ranges(year, logistic, param, sigma)
    plt.figure()
    # plotting the graph
    plt.plot(df_crl["Years"], df_crl["Australia"], label="Australia")
# plotting the forecast in data
    plt.plot(year, forecast, label="forecast", color="brown")
# showing the error ranges in plot
    plt.fill_between(year, low, up, color="violet", alpha=0.8)
    plt.title("Error Ranges for Australia", fontsize=20)
    plt.xlabel("year")
    plt.ylabel("Australia")
    plt.legend(loc='upper left')
    plt.savefig("errorrange" + '.png')
    plt.show()
    return


# Reading data source for clustering and fitting
clusterdata = pd.read_csv("hectr.csv", skiprows=4, index_col=False)
perarable = pd.read_csv("arableland.csv", skiprows=4, index_col=False)
fittingdata = datafilter(read_data('crlland.csv'))
hectarable = pd.read_csv("hectr.csv", skiprows=4, index_col=False)


# calling clustering function along and passing arguments
df_co3 = clusterdata[["1970", "1980", "1990", "2000", "2010", "2020"]]
callingcluster(df_co3, "Scatter matrix1", "1970", "2020", "Arable land (hectares per person)",
               "Arable land(1970)", "Arable land(2020)", "cluster1")

# calling clustering function for second cluster
# merging data from two dataset and combing to single dataset for clustering
hectarable = hectarable[hectarable["2020"].notna()]
perarable = perarable.dropna(subset=["2020"])
# copying required fields from each dataset
df_hec2020 = hectarable[["Country Name", "Country Code", "2020"]].copy()
df_per2020 = perarable[["Country Name", "Country Code", "2020"]].copy()
# merging two dataset
df_2020 = pd.merge(df_hec2020, df_per2020, on="Country Name", how="outer")
df_2020.to_csv("agr_for2020.csv")
df_2020 = df_2020.dropna()
# renaming the columns
df_2020 = df_2020.rename(
    columns={"2020_x": "Arable hectare", "2020_y": "Arable percentage"})
callingcluster(df_2020, "Scatter matrix2", "Arable hectare", "Arable percentage", "Arable land in 2020",
               "Arable land(hectares per person)", "Arable land(% of land area)", "cluster2")

# calling function for fitting and passing parameter
fitting(fittingdata)
