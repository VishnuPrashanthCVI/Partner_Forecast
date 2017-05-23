import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm as plf
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer
import pickle as pkl

def load_data():
	source = os.listdir('.')
	datafiles = []
	filenames = []
	data = {}

	for names in source:
		if names.endswith('.pkl'):
			datafiles.append(names)
			filenames.append(names.rstrip('.pkl'))
	filenames.sort()
	datafiles.sort()

	for filename in filenames:
		file = filename + '.pkl'
		f = open(file,'rb')
		data[filename] = pkl.load(f)
		f.close()
	return data, filenames

#form data dictionary into separate files
def teamsat_data(data=data, filenames=filenames):
	teamsat_data = pd.DataFrame()
	for name in filenames[12:24]:
			teamsat_data = pd.concat([teamsat_data, data[name]], axis=0, ignore_index=True)
	return teamsat_data

def partweb_data(data=data, filenames=filenames):
	partweb_data = pd.DataFrame()
	for name in filenames[24:36]:
			partweb_data = pd.concat([partweb_data, data[name]], axis=0, ignore_index=True)
	return partweb_data

def partinfo_data(data=data, filenames=filenames):
	partinfo_data = pd.DataFrame()
	for name in filenames[36:48]:
			partinfo_data = pd.concat([partinfo_data, data[name]], axis=0, ignore_index=True)
	return partinfo_data

def rating_data(data=data, filenames=filenames):
	rating_data = pd.DataFrame()
	for name in filenames[:12]:
			rating_data = pd.concat([rating_data, data[name]], axis=0, ignore_index=True)
	return rating_data

def partsat_data(data=data, filenames=filenames):
	partsat_data = pd.DataFrame()
	for name in filenames[48:60]:
			partsat_data = pd.concat([partsat_data, data[name]], axis=0, ignore_index=True)
	return partsat_data

def partwa_data(data=data, filenames=filenames):
	partsa_data = pd.DataFrame()
	for name in filenames[60:]:
			partsa_data = pd.concat([partsa_data, data[name]], axis=0, ignore_index=True)
	return partwa_data

#create dataframes for responses


#calculation rmse
def impute_nan(ds):
	ds = np.array(ds)
	imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
	X = imp.fit_transform(ds)
	Y = imp.fit_transform(ds)
	S = Y.shape[0]
	Y[:,-1]=np.nan
	Y = imp.fit_transform(Y)
	R = ((((X[:,-1]-Y[:,-1])**2).sum())/S)**.5
	return X, R

#create factor program training dictionary
def melt_train_dict(dff):
	X = []
	y=[]
	dff['IDX'] = range(dff.shape[0])
	dfm = pd.melt(dff,id_vars=['IDX'])
	not_null_vec = pd.notnull(dfm.value)
	df_data = dfm[not_null_vec]
	for i in range(len(df_data)):
		X.append({'response_id':str(df_data.iloc[i,0]),'ques_id':str(df_data.iloc[i,1])})
		y.append(float(df_data.iloc[i,2]))
	return X, y, df_data

#create factor program prediction dictionary
def melt_pred_dict(dff):
	Xpred=[]
	dff.IDX = range(dff.shape[0])
	dfm = pd.melt(dff,id_vars=['IDX'])
	null_vec = pd.isnull(dfm.value)
	df_predict = dfm[null_vec]
	for i in range(len(df_predict)):
		Xpred.append({'response_id': str(df_predict.iloc[i,0]),'ques_id': str(df_predict.iloc[i,1])})
	return Xpred, df_predict

#vectorize dictionaries
def vectorize(Xtrain,Xtest,Xpred):
	from sklearn.feature_extraction import DictVectorizer
	v= DictVectorizer()
	X_train = v.fit_transform(Xtrain)
	X_test = v.transform(Xtest)
	X_pred = v.transform(Xpred)
	return X_train,X_test,X_pred


def merge_data(df_predict, y_pred, df_data, qtr):
	df_predict.drop('value',axis=1,inplace=True)
	df_predict['value']=y_pred
	dfx = pd.concat([df_data,df_predict])
	dfx.sort_index(inplace=True)
	dfdata = dfx.pivot_table(index = 'IDX', columns=['variable'],values='value')
	dfdata.fillna(method='ffill', inplace = True)
	dfdata['Qtr'] = qtr
	return dfdata

if __name__ == '__main__':
	pts = teamsat_data()#partner satisfaction with acct team
	pws = partweb_data()#partner web usage and evaluation
	pwa = partwa_data()#partner sales activity
	psat = partsat_data()#partner info and satisfaction
	rate = rating_data()#company evaluation
	pinfo = partinfo_data()#basic partner information
