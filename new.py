# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:21:17 2023

@author: mypc
"""
import pandas as pd
import scipy.optimize as opt
import numpy as np
import errors as err
import matplotlib.pyplot as plt

def read_data(filename):
    '''
    Reading and performing file manipulation is done here.
    The file passed from the function call is received in variable named filename.
    '''
    dframe = pd.read_csv(filename, skiprows=4)
    return dframe



def datafilter(dframe):
    '''
    The arguments passed are received in parameters called dframe, col,value,coun and yr.
    Grouping of data is done using the function groupby.
    Indexes are reset for new dataframe.
    '''
    df_pop=dframe.loc[:,'Country Name':'2020']
    df_pop=df_pop.drop(['Country Code','Indicator Name','Indicator Code','1960'],axis=1)
    df_pop_T=df_pop.transpose()
    df_pop_T=df_pop_T.rename({'Country Name':'Years'})
    df_pop_T=df_pop_T.reset_index(drop=False)
    df_pop_T.columns=df_pop_T.iloc[0,:]
    df_pop_T=df_pop_T.drop(0)
    df_pop_T['Australia']=df_pop_T['Australia'].astype(float)
    df_pop_T['Years']=df_pop_T['Years'].astype(float)
    df_pop_T.to_csv("data123.csv")
    print( df_pop_T['Australia'].dtypes)
    return df_pop_T

def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


df_crl=datafilter(read_data('crlland.csv'))


param, covar = opt.curve_fit(logistic, df_crl["Years"], df_crl["Australia"],p0=(1.2e12, 0.03, 1961.0))
sigma = np.sqrt(np.diag(covar))
df_crl["fit"] = logistic(df_crl["Years"], *param)
df_crl.plot("Years", ["Australia", "fit"])
plt.show()

param, covar = opt.curve_fit(logistic, df_crl["Years"], df_crl["Australia"],p0=(1.2e12, 0.03, 1961.0))
sigma = np.sqrt(np.diag(covar))
year = np.arange(1960, 2040)
forecast = logistic(year, *param)
plt.figure()
plt.plot(df_crl["Years"], df_crl["Australia"], label="Australia")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("Australia")
plt.legend()
plt.show()

low, up = err.err_ranges(year, logistic, param, sigma)
plt.figure()
plt.plot(df_crl["Years"], df_crl["Australia"], label="Australia")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("Australia")
plt.legend()
plt.show()