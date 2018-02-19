import os as os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm as plf
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split

seed = 73
np.random.seed(73)

'''	Prepare data for the LSTM model
	Requires removing label data, converting to numpy arrays, shifting for lag and targets, and replacing Nan using matrix factorization
	Includes methods for use in prediction program by importing the class and methods'''

class PrepDataLSTM():
	def __init__(self):
		pass

	def split_data(self, data, seed = 73, ts=.1,k=-1,l=1):
		'''Splits dataframe to feature and prediction datasets.  Sorts by ID for shift function loop.  Splits training set as most recent sixteen quarters and test set as last quarter.

		Inputs:  data to be split
		Outputs: split and framed data that requires conversion to array for LSTM fit'''
		#confirm the sort for individual time series by ID
		data = data.sort_values(by = ['ID', 'Date'], axis = 0).reset_index(drop=True)
		#drop label columns and leave ID and Date in dataframe
		#ID left in dataframe to iterate through all dealer id's
		drops = data.columns[1:9]
		X = data.drop(drops,axis=1,inplace=False).reset_index(drop=True)
		#compute mean of input features for feature reduction
		drops = X.columns[1:-2]
		X['Mean'] = X.loc[:,drops].mean(axis=1)
		#X_pred['Mean'] = X_pred.loc[:,drops].mean(axis=1)
		cols = ['ID', 'Mean', 'Rev', 'Date']
		X = X.loc[:,cols]
		#slice columns for scaling
		rev = np.array(X.loc[:,'Rev']).reshape(-1,1)
		mean = np.array(X.loc[:,'Mean']).reshape(-1,1)
		#default activation for lstm is tanh rnage = -1,1
		scalerrev = MinMaxScaler(feature_range=(-1,1))
		scalermean = MinMaxScaler(feature_range=(-1,1))
		#scalery = MinMaxScaler(feature_range=(-1,1))
		revscaled = scalerrev.fit_transform(rev)
		meanscaled = scalermean.fit_transform(mean)
		#concatenate arrays on y axis
		Xns = np.concatenate((meanscaled,revscaled),axis=1)
		#comvert scaled arrays into dataframes
		Xs = pd.DataFrame(data=Xns,columns=cols[1:3])
		Xs['ID'] = X.ID
		Xs['Date'] = data.Date
		Xs = Xs.loc[:,cols].reset_index(drop=True)
		#return dataframes with scaled data
		return  Xs, scalerrev

	def shift_data(self,data, n_in, n_out):
		'''Shifts data to form time shifts for LSTM model
			Input:  data to be repetitively shifted
					n_in number of backward time shifts to be appended
					n_out number of forward timeshifts to be appended

			Output: dataframe of shifted data with generic variable names - replaced columns names with generic identifiers'''
		n_vars = 1 if type(data) is list else data.shape[1]
		cols, names = [],[]
		# input sequence (t-n, ... t-1)
		for i in range(n_in, 0, -1):
			cols.append(data.shift(i))
			names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
		# forecast sequence (t, t+1, ... t+n)
		for i in range(0, n_out):
			cols.append(data.shift(-i))
			if i == 0:
				names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
			else:
				names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
		# put it all together
		ds = pd.concat(cols, axis=1)
		ds.columns = names
		return ds

	def melt_dict(self, data):
		''' Method to add time scale, melt into dictionary of questions and responses for use in matrix factorization
		Input: dataframe of questions and responses with NaN
		Output: dataframe of responses and ids in vector'''

		#define new lists
		X = []
		y = []
		Xpred = []
		#add T paramter as time periods
		t = list(range(0,data.shape[0]))
		data['T'] = t
		#melt into single dictionary vector on T as key
		data_melt = pd.melt(data,id_vars=['T'])
		#identify training set using notnull on melt vector
		not_null_vec = pd.notnull(data_melt.value)
		#identify prediction set using isnull on melt vector
		null_vec = pd.isnull(data_melt.value)
		#create training and prediction vectors
		data_train = data_melt[not_null_vec]
		data_predict = data_melt[null_vec]
		#create dictionaries of each column and rank entered
		for i in range(len(data_train)):
			X.append({'response_id':str(data_train.iloc[i,0]),'ques_id':str(data_train.iloc[i,1])})
			y.append(float(data_train.iloc[i,2]))

		for i in range(len(data_predict)):
			Xpred.append({'response_id': str(data_predict.iloc[i,0]),'ques_id': str(data_predict.iloc[i,1])})
		return X, y, Xpred, data_train, data_predict

	def apply_fm(self, X, y, Xpred, seed=73):
		'''Vectorizes dictionaries of questions and responses and applies matrix factorization to Nan replacement
		input:  training and test data
		output: y predicted replacing Nan in data'''

		#create test splits for mse calculation
		Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=seed, train_size = .9)
		#vectorize the dictionary
		v= DictVectorizer()
		X_train = v.fit_transform(Xtrain)
		X_test = v.transform(Xtest)
		X_pred = v.transform(Xpred)
		#create instance of plf
		fm = plf.FM(num_factors=16, num_iter=25, verbose=True, task="regression", initial_learning_rate=0.01, learning_rate_schedule="optimal")
		#run training data through matrix factorization
		fm.fit(X_train,ytrain)
		#make predictions on test set
		y_test = fm.predict(X_test)
		#compute mse on test set
		error = mse(ytest, y_test)
		#compute predicted values for nan set
		y_pred = fm.predict(X_pred)
		return y_pred, error

	def merge_data(self, data_predict, y_pred, data_train):
		#replace values in vector with predicted values
		data_predict = data_predict.drop(['value'],axis=1)
		data_predict.loc[:,'value']=y_pred
		#concatenate vectors vertically and sort by index
		dfx = pd.concat([data_train,data_predict])
		#dfx = dfx.sort_index()
		#pivot vector back into data table
		dfx = dfx.pivot_table(index = ['T'], columns=['variable'])
		#remove multi index
		dfx = pd.DataFrame(dfx.values, index=dfx.index, columns=dfx.columns.levels[1])
		#remove columns names 'variable'
		dfx.columns.names = [None]
		dfx.reset_index(drop=True)
		return dfx

	def shape_data(self, X):
		'''Converts and shapes data into 3d numpy array required by LSTM model - drops dates and ID
		Inputs:  Dataframe of all input data including data to be predicted, Date and ID
		Outputs:  training and test features; training and test ouput targets, all 4 in 3d array format for single time step per epoch'''

		ycols = X.columns[-2:]
		dropcols = X.columns[-3:]
		y = X.loc[:,ycols]
		Xtest = X[X['Date'] == '2014-12-01']
		ytest = y[y['Date'] == '2014-12-01']
		X = X.drop(dropcols,axis=1).values
		Xtest = Xtest.drop(dropcols,axis=1).values
		X = X.reshape(X.shape[0],1,X.shape[1])
		Xtest = Xtest.reshape(Xtest.shape[0],1,Xtest.shape[1])
		y = y.drop('Date', axis=1).values
		ytest = ytest.drop('Date',axis=1).values
		return X,Xtest,y,ytest
