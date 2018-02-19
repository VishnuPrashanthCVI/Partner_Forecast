import os as os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

class PrepDataNN():
	def __init__(self):
		pass
	def prep_data_index(self, seed):
		with open('Partner_Quarterly_Revenue.pkl', 'rb') as f:
			dpr = pkl.load(f)
		#load partner data as df
		with open('partner_data.pkl', 'rb') as f:
			dpd = pkl.load(f)
		#sort quarterly revenue by date and id
		dpr = dpr.sort_values(by=['Date','ID'])
		#shift revenue by one quarter for seasonal index
		push = dpr[dpr.Date<'2011-06-01'].shape[0]
		dpr['PQR'] = dpr.Revenue.shift(push)
		dpr['Seas_Ind']=dpr.Revenue/dpr.PQR
		#truncate revenue data for data set
		dpi = dpr[dpr.Date>'2011-03-01']
		dpi = dpi[dpi.Date < '2014-12-01']
		#create revenue data
		ydata = dpi.Seas_Ind
		y_pred = dpr.Seas_Ind[dpr.Date == '2014-12-01']
		#create independent data
		#add TI as index for slicing
		s1 = [str(yr) for yr in dpd.Yr]
		s2 = [str(qtr) for qtr in dpd.Qtr]
		s3 = []
		for i in range(len(s1)):
		    s3.append(s1[i]+s2[i])
		dpd['TI'] = s3
		#columns to drop from multivariate projections
		cols = list(dpd.columns[:9])
		cols.append(dpd.columns[-1])
		#quarterly data to predict revenue
		x_pred = dpd[dpd.TI=='44'].drop(cols,axis=1,inplace=False).reset_index(drop=True)
				#quarterly data to run analysis
		xdata = dpd[dpd.TI<'44'].reset_index(drop=True)
		xdata = xdata[xdata.TI>'11'].reset_index(drop=True).drop( cols,axis=1, inplace=False)
		#make train test split
		y_train = y_train.values
		x_train = x_train.values
		y_test = y_test.values
		x_test = x_test.values
		#scale all factors for relu activation
		scaler=StandardScaler()
		x_train = scaler.fit_transform(x_train)
		x_test = scaler.fit_transform(x_test)
		#y_train = scaler.fit_transform(y_train)
		#y_test = scaler.fit_transform(y_test)

		return x_train, y_train, x_test, y_test
