import os as os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from fbprophet import Prophet
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import math as ma
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import sgd,rmsprop,adam
from keras.initializers import RandomNormal, RandomUniform
from keras.losses import mean_squared_error, mean_absolute_error
import keras
from keras.callbacks import CSVLogger

from time_series import TimeSeries

seed = 73
np.random.seed(seed)

#load and strip pickle file
with open('Partner_Monthly_Revenue.pkl', 'rb') as f:
	data = pkl.load(f)
#read in monthly revenue by partner and month
#initiate time series class
ts = TimeSeries(data)

#use times series to predict 6 months time frame
dt = pd.DataFrame()
dts = pd.DataFrame()

#forecast for each id and monthly series
for id in range(1, data.ID.max() + 1):
	df=ts.time_data(id)
	dp=ts.fit_time(df)
	idx = [id for j in range(0,dp.shape[0])]
	dts['ID'] = idx
	dts['Date'] = dp.ds
	dts['Act_Revenue'] = data.Revenue
	dts['Pred_Revenue']=np.exp(dp.yhat)
	dts['Low_Revenue'] = np.exp(dp.yhat_lower)
	dts['High_Revenue'] = np.exp(dp.yhat_upper)
	dts['Trend'] = np.exp(dp.trend)
	dt = pd.concat([dt,dts],axis=0,ignore_index=True)

#result in dt is now 54 periods long
#calculate mean and error into dictionary
time_error = ts.time_error(dt)

#save results
filestr = 'Time_Series_Predict.pkl'
with open(filestr, 'wb') as f:
	pkl.dump(dts, f, -1)

#save time series error
filestr = 'Time_Series_Error.pkl'
with open(filestr, 'wb') as f:
	pkl.dump(time_error, f, -1)
