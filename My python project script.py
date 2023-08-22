# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Python script for my AIT 622 Big Data Analytics Project

# Run & compare the performances of many different common/popular
# statistical/machine learning algorithms on 5 sets of stock market data
# for 5 different years in terms of how well they are able to forecast.

import numpy as np
import pandas as pd
import os as os



os.getcwd()
os.chdir('/AIT 622 - Determining Needs for Complex Big Data Systems/AIT 622 Individual Project')


data2014 = pd.read_csv("2014_Financial_Data.csv", header = 1)
data2015 = pd.read_csv("2015_Financial_Data.csv", header = 1)
data2016 = pd.read_csv("2016_Financial_Data.csv", header = 1)
data2017 = pd.read_csv("2017_Financial_Data.csv", header = 1)
data2018 = pd.read_csv("2018_Financial_Data.csv", header = 1)



##### data cleaning, wrangling, and pre-processing:
## remove all predictors with either zero or near zero variance
from sklearn.feature_selection import VarianceThreshold
IVs = VarianceThreshold()
IVs.fit_transform(data2014)



