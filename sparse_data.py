import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm as plf
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer


class SparseData:
	def __init__(self, data):
		self.data=data
	#check if nulls exist
	def number_nulls(self):
		no_nans={}
		for col in self.data.columns:
			no_nans[col] = (self.data[col].isnull().sum())
		return no_nans

		#use standard impute with means on columns
	def impute_nan(self):

		imputed_data = pd.DataFrame()

		data_imp =self.data.sort_values(by = ['Yr','Qtr', 'Response_Number'], inplace=False)

		drop_resp_cols = list(data_imp.columns[11::])
		reg_data=data_imp.drop(drop_resp_cols, axis =1)

		drop_reg_cols=list(data_imp.columns[0:11])
		resp_data = data_imp.drop(drop_reg_cols, axis = 1)

		impute = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
		imputed_resp_data = pd.DataFrame(impute.fit_transform(resp_data))
		imputed_resp_data.columns = drop_resp_cols

		imputed_data = pd.concat([reg_data,imputed_resp_data], axis = 1, ignore_index = False)
		imputed_data.sort_values(by=['Response_Number'], inplace = True)

		return imputed_data

	#implement matrix facorization
	#create factor program  dictionaries
	def melt_dict(self):
		data_resp = pd.DataFrame()
		data_time = pd.DataFrame()
		#sort data into yr and qtr to maintain time factor and be working data
		data_time = self.data.sort_values(by = ['Yr','Qtr', 'Response_Number'], ascending=True)
		data_cols = data_time.columns

		#define new lists
		X = []
		y = []
		Xpred = []

		#remove registration data and all string entries
		drop_reg_cols= list(data_time.columns[0:11])
		data_resp = data_time.drop(drop_reg_cols, axis = 1)

		#create dataframe of registration data
		drop_cols=list(data_time.columns[11::])

		#registrion data to merge with filled response data
		data_reg = data_time.drop(drop_cols, axis = 1)

		#inset continuous index for melt keys
		data_reg['IDX']=range(1, 1 + data_time.shape[0])
		#move index to first column
		data_reg_cols = [data_reg.columns[-1]] + list(data_reg.columns[0:-1])
		data_reg= data_reg.ix[:,data_reg_cols]

		#inset continuous index for melt keys
		data_resp['IDX']=range(1, 1 + data_time.shape[0])
		#move index to first column
		data_resp_cols = [data_resp.columns[-1]] + list(data_resp.columns[0:-1])
		data_resp = data_resp.ix[:,data_resp_cols]

		#melt into single dictionary vector on IDX as key
		data_melt = pd.melt(data_resp,id_vars=['IDX'])

		#identify training set using notnull on melt vector
		not_null_vec = pd.notnull(data_melt.value)

		#identify prediction set using isnull on melt vector
		null_vec = pd.isnull(data_melt.value)

		#create training and prediction vectors
		data_train = data_melt[not_null_vec]
		data_predict = data_melt[null_vec]

		#convert data vectors into dictionaries
		for i in range(len(data_train)):
			X.append({'response_id':str(data_train.iloc[i,0]),'ques_id':str(data_train.iloc[i,1])})
			y.append(float(data_train.iloc[i,2]))

		for i in range(len(data_predict)):
			Xpred.append({'response_id': str(data_predict.iloc[i,0]),'ques_id': str(data_predict.iloc[i,1])})

		return X, y, Xpred, data_train, data_predict, data_resp_cols, data_reg, data_resp

	def apply_fm(self, X, y, Xpred):

		#create test splits for mse calculation
		Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=73, train_size = .9)
		#vecotrize the dictionary
		v= DictVectorizer()
		X_train = v.fit_transform(Xtrain)
		X_test = v.transform(Xtest)
		X_pred = v.transform(Xpred)
		#create instance of plf
		fm = plf.FM(num_factors=15, num_iter=25, verbose=True, task="regression", initial_learning_rate=0.02, learning_rate_schedule="optimal")
		#run training data through matrix factorization
		fm.fit(X_train,ytrain)
		#make predictions on test set
		y_test = fm.predict(X_test)
		#compute mse on test set
		mse = mean_squared_error(ytest, y_test)
		#compute predicted values for nan set
		y_pred = fm.predict(X_pred)

		return y_pred, mse

	def merge_data(self, data_predict, y_pred, data_train, data_reg):
		#replace values in vector with predicted values
		data_predict.drop('value',axis=1,inplace=True)
		#Y = []
		#for y in y_pred:
		#	Y.append(int(round(y)))
		data_predict['value']=y_pred
		#concatenate vectors vertically
		dfx = pd.concat([data_train,data_predict])
		#sort by index back into original vector
		dfx = dfx.sort_index()
		#pivot vector back into data table
		dfx = dfx.pivot_table(index = ['IDX'], columns=['variable'])
		#remove multi index
		dfp = pd.DataFrame(dfx.values, index=dfx.index, columns=dfx.columns.levels[1])
		#remove columns names 'variable'
		dfp.columns.names = [None]
		#start index at 0
		dfp = dfp.reset_index()
		#prepare dfp for groupby operation to remove multiple entries by ID
		dfp.drop('IDX', axis = 1, inplace = True)
		#add Yr Qtr ID columns to dfp from data_reg
		dfp['Yr'] = data_reg.Yr
		dfp['Qtr'] = data_reg.Qtr
		dfp['ID'] = data_reg.ID
		#rollup each Yr Qtr ID by mean
		dfp = dfp.groupby(['Yr', 'Qtr', 'ID']).mean()
		#remove multiple index of Yr, Qtr and ID
		dfp = dfp.reset_index()
		#remove redundant response info
		data_reg.drop(['Response_Number', 'Respondent', 'IDX'], axis =1, inplace = True)
		#drop duplicate responses by yr,qtr and id
		dfr = data_reg.drop_duplicates(['Yr','Qtr','ID']).reset_index(drop=True)
		#merge registration data with prepared data
		df = pd.merge(dfr, dfp, on = ['Yr','Qtr','ID'])

		return df
