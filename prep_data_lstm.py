import os as os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

class PrepData():
	def __init__(self):
		pass
	def prep_data(self,dpr,dpd):
		#fix random seed for reproducibilitypc
		dpr = dpr.sort_values(by=['Date','ID'])
		#shift prior period revenue into current period as time predictor
		dpr['Prior_Rev'] = dpr.Revenue.shift(300)
		#remove first qtr data to remove NAN's formed by shift
		dpr = dpr[dpr.Date>'2011-03-01']
		#create revenue data to be predicted
		y_test = dpr.Revenue[dpr.Date>'2014-09-01'].reset_index(drop=True)
		x_test_prior =  dpr.Prior_Rev[dpr.Date>'2014-09-01'].reset_index(drop=True)
		#create revenue data for training dataset
		y_train = dpr.Revenue[dpr.Date<'2014-12-01'].reset_index(drop=True)
		x_train_prior = dpr.Prior_Rev[dpr.Date<'2014-12-01'].reset_index(drop=True)
		#transform partner data for training and test data
		#remove first and last quarter to match revenue data
		#add TI as index for slicing
		s1 = list(map(str, dpd.Yr))
		s2 = list(map(str,dpd.Qtr))
		s3 = []
		for i in range(len(s1)):
		    s3.append(s1[i]+s2[i])
		dpd['TI']=s3
		#quarterly data to predict revenue
		cols = ['TI','ID','Yr','Qtr']
		x_test = dpd[dpd.TI=='44'].drop(cols,axis=1,inplace=False).reset_index(drop=True)
		#quarterly data to run lstm
		x_train = dpd[dpd.TI>'11'][dpd.TI<'44'].drop( cols,axis=1, inplace=False).reset_index(drop=True)
		#merge prior revenue into data set
		#x_train['Prior_Rev'] = x_train_prior
		#x_test['Prior_Rev'] = x_test_prior

		#lebel encode and hot encode classifications
		cols = x_train.columns[:6]
		#encode dataset and predict set
		#for col in cols:
		#	c = x_train[col]
		#	encoder=LabelEncoder()
		#	encoder.fit(c)
		#	encoded_c = encoder.transform(c)
		#	x_train[col] = encoded_c
		#	d = x_test[col]
		#	encoder.fit(d)
		#	encoded_d = encoder.transform(d)
		#	x_test[col]=encoded_d
		#drop label encoded columns if fit is weak
		x_train = x_train.drop(cols,inplace=False,axis=1).reset_index(drop=True)
		x_test = x_test.drop(cols,inplace=False,axis=1).reset_index(drop=True)
		#make names explicit and convert to array
		y_train = y_train.values.reshape(-1,1)
		x_train = x_train.values
		y_test = y_test.values.reshape(-1,1)
		x_test = x_test.values
		#scale all factors for relu activation
		scaler=MinMaxScaler()
		x_train = scaler.fit_transform(x_train)
		x_test = scaler.fit_transform(x_test)
		y_test = scaler.fit_transform(y_test)
		y_train = scaler.fit_transform(y_train)

		return x_train, y_train, x_test, y_test
