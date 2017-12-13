import os as os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

class PrepData():
	def __init__(self):
		pass
	def prep_data(self,dpr,dpd):
		#sort quarterly revenue by data and id
		dpr = dpr.sort_values(by=['Date','ID'])
		#create revenue data to be predicted
		y_test = dpr.Revenue[dpr.Date == '2014-12-01'].reset_index(drop=True)
		#create revenue data for training dataset
		y_train = dpr.Revenue[dpr.Date<'2014-12-01'].reset_index(drop=True)
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
		x_test = dpd[dpd.TI=='44'].drop(cols,axis=1,inplace=False).reset_index(drop=True)
		#quarterly data to run analysis
		x_train = dpd[dpd.TI<'44'].drop( cols,axis=1, inplace=False).reset_index(drop=True)
		#make names explicit and convert serues to array with reshape
		y_train_values = y_train.values.reshape(-1,1)
		x_train_values = x_train.values
		y_test_values = y_test.values.reshape(-1,1)
		x_test_values = x_test.values
		#scale all factors for relu activation
		scaler=MinMaxScaler()
		x_train_scaled = scaler.fit_transform(x_train_values)
		x_test_scaled = scaler.fit_transform(x_test_values)
		y_test_scaled = scaler.fit_transform(y_test_values)
		y_train_scaled = scaler.fit_transform(y_train_values)

		return x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled

	def inv_trans(self,y,dpr):
		y_range = dpr.Revenue[dpr.Date>'2014-09-01'].reset_index(drop=True).values.reshape(-1,1)
		max = y_range.max()
		min = y_range.min()
		y = scaler.inverse_transform(y)
		return y
