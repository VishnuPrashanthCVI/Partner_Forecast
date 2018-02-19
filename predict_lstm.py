import os as os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
#from fbprophet import Prophet
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import math as ma
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization,LSTM
from keras.optimizers import sgd,rmsprop,adam
from keras.initializers import RandomNormal, RandomUniform
from keras.losses import mean_squared_error, mean_absolute_error
from keras.callbacks import CSVLogger

from prep_data_LSTM import PrepDataLSTM
from lstm_model import ModelDataLSTM
#create instance
pdl = PrepDataLSTM()
lstm = ModelDataLSTM()
#set seed for reproducibility
seed = 73
np.random.seed(73)
#fetch data
with open('Part_Qtr_Rev_Data.pkl', 'rb') as f:
	data = pkl.load(f)
#function to shift data by ID - set n_out to 0 to remove current period feature set
def shift_fit_id(data,n_in=4,n_out=1):
	'''Iterates through ID identities to create shifted dataframe.  NaN created by shift are replaced using the matrix factorization scheme during each ID loop.  The shifted and filled data is concatenated to form complete dataframe including all ID's.
	Input:  data sorted by ID and Date including feature set and variable to be predicted, number of periods to be included in dataframe.  n_out = 1 includes current period in feature set.
	Output:  unscaled and shifted with mean of feature set in data and errors from matrix factorization computation.
	Note:  If prior periods of prediction variable to be included in shifted data for time series prediction then n_out must be set to 0'''
	dfs = pd.DataFrame()
	errors=[]
	ID = []
	dates =[]
	#shift and concatenate data
	for id in range(data.ID.min(),data.ID.max()+1):
		#split out dealers by id
		df = data[data['ID'] == id]
		#create ID vector to be restored at end
		ID.extend(df['ID'])
		#create dates vector to be restored at end
		dates.extend(df['Date'])
		#drop ID so not shifted with data
		df = df.drop(['ID','Date'],axis=1)
		ds = pdl.shift_data(df,n_in,n_out)
		cols = ds.columns
		#the melt, dict and pivot operations scrambles columns
		X, y, Xpred, data_train, data_predict=pdl.melt_dict(ds)
		y_pred, error = pdl.apply_fm(X, y, Xpred)
		errors.append(error)
		dfi = pdl.merge_data(data_predict, y_pred, data_train)
		#restore column order
		dfi = dfi.loc[:,cols]
		#concat by id
		dfs = pd.concat([dfs,dfi]).reset_index(drop=True)
	#restore ID column
	col = ['ID']
	col.extend(dfs.columns)
	col.append('Date')
	dfs['ID'] = ID
	dfs['Date'] = dates
	dfs = dfs.loc[:,col].reset_index(drop=True)
	return dfs, errors

#class method to remove label data, scale, and reduce dimensions
X, scaler = pdl.split_data(data)
#shift four periods, replace NaN and concatenate data by ID
n_in,n_out=4,1
X, errors = shift_fit_id(X,n_in=n_in,n_out=n_out)
# saving prepared files for later use
X.to_csv('LSTM_data.csv',index=False)
#apply LSTM model by ID
#for debugging - data after shift, fill and scale data and shift_data
#X = pd.read_csv('LSTM_data.csv', sep = ',')
errors=[]
ID = []
cols = []
Yp = Ytp = pd.DataFrame()
for id in range(X.ID.min(),X.ID.max()+1):
	#clear previoius results
	df = yp = ytp = pd.DataFrame()
	#separate out single id
	df = X[X['ID'] == id]
	ID.extend(df.ID)
	#shape data into 3d numpy arrays
	Xtrain,Xtest,ytrain,ytest = pdl.shape_data(df.drop('ID',axis=1))
	#call model in lstm class
	model = lstm.lstm_model(Xtrain,b=1,n=24,o=1)
	#fit model and build prediction vectors
	yp,ytp = lstm.lstm_fit(model,Xtrain,Xtest,ytrain)
	#rescale predictions
	yp = scaler.inverse_transform(yp)
	ytp = scaler.inverse_transform(ytp)
	#reform dataframes from arrays
	yp = pd.DataFrame(data=yp,columns=['Rev'])
	ytp = pd.DataFrame(data=ytp,columns=['Rev'])
	Yp = pd.concat([Yp,yp])
	Ytp = pd.concat([Ytp,ytp])

#append ID,Date columns
Yp['ID'] = ID
Ytp['ID'] = range(X.ID.min(),X.ID.max()+1)
Yp['Date'] = X.Date
Ytp['Date'] = X[X['Date'] == '2014-12-01'].Date
#inverse transform scaled revenue columns from LSTM data used in fit
varname = X.columns[-2]
revenue = scaler.inverse_transform(X[varname].values.reshape(-1,1))
Yp['Act_Rev'] = revenue
revenue = Yp[Yp['Date'] == '2014-12-01'].Act_Rev
Ytp['Act_Rev'] = revenue.values.reshape(-1,1)
#order columns
cols = ['ID','Rev','Act_Rev','Date']
Yp = Yp.loc[:,cols]
Ytp = Ytp.loc[:,cols]
#save results
Yp.to_csv('LSTM_predictions.csv', index=False)
Ytp.to_csv('LSTM_predictions_lastqtr.csv', index=False)

mean_sq_error = mse(Yp.Act_Rev,Yp.Rev)
mean_abs_error = mae(Yp.Act_Rev,Yp.Rev)
mean_percent_error = (abs(Yp.Act_Rev - Yp.Rev)/Yp.Act_Rev).mean()
error = [mean_sq_error,mean_abs_error,mean_percent_error*100]
errors = pd.DataFrame()
errors['Mean_Metric'] = ['Squared','Absolute','Percentage']
errors['Error'] = error
errors.to_csv('LSTM_error.csv', index=False)
'''for debugging - data after prep_data and shift_data
X = pd.read_csv('LSTM_data.csv', sep = ',')
'''
