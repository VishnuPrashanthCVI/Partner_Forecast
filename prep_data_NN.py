import os as os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

class PrepDataNN():
	def __init__(self,data):
		self.data = data
	def prep_data_NN(self, seed = 73, ts = .1):
		#fetch data
		#with open('Part_Qtr_Rev_Data.pkl', 'rb') as f:
		#	data = pkl.load(f)
		#confirm the sort for revenue offset for seasonal index and remove self for simplicity
		data = self.data.sort_values(by = ['Date', 'ID'], axis = 0)
		#shift revenue by one quarter for seasonal index
		push = data[data.Date < '2012-03-01'].shape[0]
		#shift revenue data by one year
		data['PQR'] = data.Rev.shift(push)
		#compute seasonal index
		data['Seas_Ind']=data.Rev/data.PQR
		#resort for ID and date
		data = data.sort_values(by = ['ID', 'Date'])
		#truncate revenue data for data set
		dpy = data[data.Date>'2011-12-01']
		dp = dpy[dpy.Date < '2014-12-01']
		#confirm sort
		dp = dp.sort_values(by = ['ID', 'Date'], axis = 0)
		#create revenue data ordered by ID and Date
		y_data = dp.Seas_Ind.reset_index(drop=True)
		y_pred = data.Seas_Ind[data.Date == '2014-12-01'].reset_index(drop=True)
		#create independent data
		cols = list(dp.columns[:9])#label data
		cols.extend(dp.columns[-4:])#y data
		#remove columns with label or computed data
		x_data = dp.drop(cols,axis=1,inplace=False)
		#compute prediction data
		x_pred = data[data.Date == '2014-12-01']
		x_pred = x_pred.drop(cols,axis=1,inplace=False).reset_index(drop=True)
		#convert to array and scale data for relu function
		scaler=MinMaxScaler(feature_range = (0,1), copy = False)
		x_train = x_data.as_matrix()
		scaler.fit_transform(x_train)
		x_pred = x_pred.as_matrix()
		scaler.fit_transform(x_pred)
		#convert to array and reshape out of series notation
		y_train = y_data.values.reshape(-1,1)
		y_pred= y_pred.values.reshape(-1,1)

		return  x_train, y_data, x_pred, y_pred,
