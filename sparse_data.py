import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm as plf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer
import pickle as pkl

def impute_nan(data,axis=1):
	resp_data = data.drop(data.columns[:11], axis = 1)
	impute = Imputer(missing_values = 'NaN', strategy = 'mean', axis = axis)
	ird = impute.fit_transform(resp_data)
	with open ('true_partner_data.pkl','rb') as f:
		data_true = pkl.load(f)
	data_true = np.array(data_true)[:,11:].astype(float)
	n = pd.isnull(resp_data).sum().sum()
	N = data_true.shape[0]*data_true.shape[1]
	rmse = mean_squared_error(data_true,ird)*N/n
	return rmse, data_true

#implement matrix facorization
#create factor program  dictionaries
def melt_dict(data):
	#define new lists
	X = []
	y = []
	Xpred = []

	#remove registration data and retain response number
	data_resp = data.drop(data.columns[1:11], axis = 1)
	#registrion data to merge with filled response data
	data_reg = data.drop(data.columns[11::], axis = 1)
	#melt into single dictionary vector on IDX as key
	data_melt = pd.melt(data_resp,id_vars=['Response_Number'])
	#identify training set using notnull on melt vector
	not_null_vec = pd.notnull(data_melt.value)
	#identify prediction set using isnull on melt vector
	null_vec = pd.isnull(data_melt.value)
	#create training and prediction vectors
	data_train = data_melt[not_null_vec]
	data_predict = data_melt[null_vec]
	#convert data vectors into lists
	X = []
	y = []
	Xpred = []

	for i in range(len(data_train)):
		X.append({'response_id':str(data_train.iloc[i,0]),'ques_id':str(data_train.iloc[i,1])})
		y.append(float(data_train.iloc[i,2]))

	for i in range(len(data_predict)):
		Xpred.append({'response_id': str(data_predict.iloc[i,0]),'ques_id': str(data_predict.iloc[i,1])})
	return X, y, Xpred, data_train, data_predict, data_reg, data_resp

def apply_fm(X, y, Xpred):
	#create test splits for mse calculation
	Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=73, train_size = .9)
	#vectorize the dictionary
	v= DictVectorizer()
	X_train = v.fit_transform(Xtrain)
	X_test = v.transform(Xtest)
	X_pred = v.transform(Xpred)
	#create instance of plf
	fm = plf.FM(num_factors=20, num_iter=30, verbose=True, task="regression", initial_learning_rate=0.01, learning_rate_schedule="optimal")
	#run training data through matrix factorization
	fm.fit(X_train,ytrain)
	#make predictions on test set
	y_test = fm.predict(X_test)
	#compute mse on test set
	mse = mean_squared_error(ytest, y_test)
	#compute predicted values for nan set
	y_pred = fm.predict(X_pred)
	return y_pred, mse

def merge_data(data_predict, y_pred, data_train, data_reg):
	#replace values in vector with predicted values
	data_predict = data_predict.drop(['value'],axis=1)
	data_predict.loc[:,'value']=y_pred
	#concatenate vectors vertically and sort by index
	dfx = pd.concat([data_train,data_predict])
	#dfx = dfx.sort_index()
	#pivot vector back into data table
	dfx = dfx.pivot_table(index = ['Response_Number'], columns=['variable'])
	#remove multi index
	dfx = pd.DataFrame(dfx.values, index=dfx.index, columns=dfx.columns.levels[1])
	#remove columns names 'variable'
	dfx.columns.names = [None]
	dfx['Response_Number'] = dfx.index
	df = pd.merge(data_reg,dfx,on = ['Response_Number'])
	dfp = df.groupby(['Yr', 'Qtr', 'ID']).mean()
	dfp = dfp.drop(['Response_Number'],axis=1)
	dfp.reset_index(inplace=True)
	data_reg.drop(['Response_Number', 'Respondent', 'Credit_Rating'], axis =1, inplace = True)
	dfr =  data_reg.drop_duplicates(['Yr','Qtr','ID']).reset_index(drop=True)
	#merge registration data with prepared data
	df = pd.merge(dfr, dfp, on = ['Yr','Qtr','ID'])
	return df

if __name__ == '__main__':

	with open ('raw_partner_data.pkl','rb') as f:
		data = pkl.load(f)
	X, y, Xpred, data_train, data_predict, data_reg, data_resp = melt_dict(data)

	#apply matrix factorization to form predictions
	y_pred, mfmse = apply_fm(X, y, Xpred)

	#merge responses with predicted responses and form filled data set dfp
	df = merge_data(data_predict, y_pred, data_train, data_reg)

	#pickle new filled data for additional analysis
	with open ('NoNan_Partner_Data.pkl','wb') as f:
		pkl.dump(df, f,-1)

	colmse, _ = impute_nan(data,axis=0)
	rowmse, dataT = impute_nan(data,axis=1)
	errors = np.array([mfmse, rowmse, colmse])
	errors = pd.DataFrame(errors).T
	errors.columns = ['FM','Row_Mean','Column_Mean']
	np.savetxt('sparse_mse.csv',errors,delimiter = ',')
