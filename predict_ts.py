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

from time_series import TimeSeries

seed = 73
np.random.seed(seed)

#load and strip pickle file
with open('Partner_Monthly_Revenue.pkl', 'rb') as f:
	data = pkl.load(f)
#read in monthly revenue by partner and month
#initiate time series class
ts = TimeSeries(data)

#use times series to predict 3 months time frame
dt,de = pd.DataFrame(),pd.DataFrame()
#forecast for each id and monthly series
for id in range(1, data.ID.max() + 1):
	#identify class and territory
	c = list(data[data.ID == id].Class)[0]
	t = list(data[data.ID == id].Territory)[0]
	#set dataframes and lists to original state for each id
	dts,dte,df,dp = pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
	#fit data to time series
	df,act=ts.time_data(id)
	dp=ts.fit_time(df)
	#calculate data for new projections
	idx = [id for j in range(0,dp.shape[0])]
	cl = [c for j in range(0,dp.shape[0])]
	tl = [t for j in range(0,dp.shape[0])]
	dts['ID'] = idx
	dts['Class'] = cl
	dts['Territory'] = tl
	dts['Date'] = dp.ds
	dts['Act'] = act.reset_index(drop=True)
	dts['Pred']=np.exp(dp.yhat)
	dts['Low'] = np.exp(dp.yhat_lower)
	dts['High'] = np.exp(dp.yhat_upper)
	dts['Trend'] = np.exp(dp.trend)

	#calculating slope
	slope_list=[0,0,0,0,0,0]
	for t in range(6,len(dts)):
		point1 = dts.Trend[t-6]
		point2 = dts.Trend[t]
		slope = (point2-point1)/6
		slope_list.append(slope)
	dts['Slope'] = slope_list

#calculating absolute change from prior quarter actual to next qtr pred
	change_list = [0,0,0,0,0,0]
	for t in range(3,len(dts)-3):
		predsum = dts.Pred[t+2]+dts.Pred[t+1]+dts.Pred[t]
		actsum  = dts.Act[t-1]+dts.Act[t-2]+dts.Act[t-3]
		change = predsum-actsum
		change_list.append(change)
	dts['Change'] = change_list

	#calculating actual error in each period
	error_list = []
	for t in range(0,len(dts)-3):
		error = abs(dts.Pred[t]-dts.Act[t])
		error_list.append(error)
	error_list.extend([0,0,0])
	dts['Error'] = error_list

	#concatenate id's with slope, error and change
	dt = pd.concat([dt,dts],axis=0,ignore_index=True)

#measure errors
dte['ID'] = idx
dte['MSE'] = mse(dts.Act[:-3],dts.Pred[:-3])
de = pd.concat([de,dte],axis=0,ignore_index=True)


#result in dt is now 54 periods long
#calculate mean and error into dictionary orig_mean,pred_mean and root_mean_error as keys
average_time_error = ts.average_time_error(dt)

#save results
filestr = 'Time_Series_Predict.pkl'
with open(filestr, 'wb') as f:
	pkl.dump(dt, f, -1)

#save time series error
filestr = 'Average_Time_Series_Error.pkl'
with open(filestr, 'wb') as f:
	pkl.dump(average_time_error, f, -1)

#save time series error
filestr = 'ID_Time_Series_Error.pkl'
with open(filestr, 'wb') as f:
	pkl.dump(de, f, -1)
