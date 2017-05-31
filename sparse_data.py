import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm as plf
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer
import pickle as pkl
import math as ma
import random as ra

def load_data():
	source = os.listdir('.')
	datafiles = []
	filenames = []
	data = {}

	for names in source:
		if names.endswith('.pkl') and names.startswith('o_'):
			filenames.append(names.rstrip('.pkl'))
	filenames.sort()


	for filename in filenames:
		file = filename + '.pkl'
		f = open(file,'rb')
		data[filename] = pkl.load(f)
		#print filename
		#print data[filename].shape
		f.close()
	return data, filenames

#form data dictionary into separate files
def teamsat_data(data, filenames):
	teamsat_data = pd.DataFrame()
	for name in filenames[12:24]:
			teamsat_data = pd.concat([teamsat_data, data[name]], axis=0, ignore_index=True)
	return teamsat_data

def partweb_data(data, filenames):
	partweb_data = pd.DataFrame()
	for name in filenames[24:36]:
			partweb_data = pd.concat([partweb_data, data[name]], axis=0, ignore_index=True)
	return partweb_data

def partinfo_data(data, filenames):
	partinfo_data = pd.DataFrame()
	for name in filenames[36:48]:
			partinfo_data = pd.concat([partinfo_data, data[name]], axis=0, ignore_index=True)
	return partinfo_data

def rating_data(data, filenames):
	rating_data = pd.DataFrame()
	for name in filenames[:12]:
			rating_data = pd.concat([rating_data, data[name]], axis=0, ignore_index=True)
	return rating_data

def partsat_data(data, filenames):
	partsat_data = pd.DataFrame()
	for name in filenames[48:60]:
			partsat_data = pd.concat([partsat_data, data[name]], axis=0, ignore_index=True)
	return partsat_data

def partwa_data(data, filenames):
	partwa_data = pd.DataFrame()
	for name in filenames[60:]:
			partwa_data = pd.concat([partwa_data, data[name]], axis=0, ignore_index=True)
	return partwa_data

#calculation rmse
def impute_nan(ds):
	C = []
	MN = []
	ds = np.array(ds)
	imp = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
	X = imp.fit_transform(ds)
	N = int(.1 * (X.shape[0] + X.shape[1]))
	i = 1
	while i <= N:
		m = ra.choice(range(X.shape[0]))
		n = ra.choice(range(X.shape[1]))
		c = ds[m,n]
		mn = np.nanmean(ds[m,:])
		if ma.isnan(c):
			i = i
		else:
			C.append(c)
			MN.append(mn)
			i = i+1
	C = np.array(C)
	MN = np.array(MN)
	R = ((((C - MN)**2).sum())/(len(C)))**.5
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
	dff['IDX'] = range(dff.shape[0])
	dfm = pd.melt(dff,id_vars=['IDX'])
	null_vec = pd.isnull(dfm.value)
	df_predict = dfm[null_vec]
	for i in range(len(df_predict)):
		Xpred.append({'response_id': str(df_predict.iloc[i,0]),'ques_id': str(df_predict.iloc[i,1])})
	return Xpred, df_predict

#vectorize dictionaries
def apply_fm(X,y,Xpred):
	Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=71)
	v= DictVectorizer()
	X_train = v.fit_transform(Xtrain)
	X_test = v.transform(Xtest)
	X_pred = v.transform(Xpred)
	fm = plf.FM(num_factors=10, num_iter=50, verbose=True, task="regression", initial_learning_rate=0.02, learning_rate_schedule="optimal")
	fm.fit(X_train,ytrain)
	y_test = fm.predict(X_test)
	mse = mean_squared_error(ytest,y_test)
	y_pred = fm.predict(X_pred)
	return y_pred, mse

def merge_data(df_predict, y_pred, df_data):
	df_predict.drop('value',axis=1,inplace=True)
	df_predict['value']=y_pred
	dfx = pd.concat([df_data,df_predict])
	dfx.sort_index(inplace=True)
	dfdata = dfx.pivot_table(index = 'IDX', columns=['variable'],values='value')
	dfdata.fillna(method='ffill', inplace = True)
	return dfdata

def merge_id(Xp,Xo):
	Xc = pd.DataFrame()
	Xn = pd.DataFrame()
	cols = Xo.columns
	Xc = Xo.drop(cols[2:], axis = 1)
	Xn = pd.concat([Xc,Xp], axis = 1, ignore_index = True)
	Xn.columns = cols
	return Xn


if __name__ == '__main__':
#load data
	data, filenames = load_data()
#identify dataframes
	pts = pd.DataFrame()
	pws = pd.DataFrame()
	psat = pd.DataFrame()
	pwa = pd.DataFrame()
	mrp = pd.DataFrame()
	pinfo = pd.DataFrame()
#create dataframes containing unique data with NaN
	pts = teamsat_data(data, filenames)#partner satisfaction with acct team
	pws = partweb_data(data, filenames)#partner web usage and evaluation
	psat = partsat_data(data, filenames)#partner satisfaction
#create dataframes for continuous and categorical data
	pwa = partwa_data(data, filenames)
	mrp = rating_data(data, filenames)
	pinfo = partinfo_data(data, filenames)
#create arrays for impute (and factorization)
	ipts = pts.drop(['ID','Resp'], axis = 1)
	ipws = pws.drop(['ID','Resp'], axis = 1)
	ipsat = psat.drop(['ID','Resp'], axis = 1)
	#calculate RMSE on imputed values
	dfiles = [ipts,ipws,ipsat]
	R = []
	for names in dfiles:
		X, r = impute_nan(names)
		R.append(r)

#create training dictionary - nn? is not null melt
	Xpts, ypts, nnpts = melt_train_dict(ipts)
	Xpws, ypws, nnpws = melt_train_dict(ipws)
	Xpsat, ypsat, nnpsat = melt_train_dict(ipsat)

#create prediction dictionary - n? is null melt
	ipts = pts.drop(['ID','Resp'], axis = 1)
	ipws = pws.drop(['ID','Resp'], axis = 1)
	ipsat = psat.drop(['ID','Resp'], axis = 1)

	Ppts, npts = melt_pred_dict(ipts)
	Ppws, npws = melt_pred_dict(ipws)
	Ppsat, npsat = melt_pred_dict(ipsat)

#run matrix factorization
	Ypts, mpts = apply_fm(Xpts, ypts, Ppts)
	Ypws, mpws = apply_fm(Xpws, ypws, Ppws)
	Ypsat, mpsat = apply_fm(Xpsat, ypsat, Ppsat)

#pivot and merge data for new data sets
	Dpts = merge_data(npts, Ypts, nnpts)
	Dpws = merge_data(npws, Ypws, nnpws)
	Dpsat = merge_data(npsat, Ypsat, nnpsat)

#replace nan data with fit data
	ptsn = merge_id(Dpts,pts)
	pwsn = merge_id(Dpws,pws)
	psatn = merge_id(Dpsat,psat)

#rollup multiple entries using apply function to groupby
#rollup on option mean
#rollup by quarter
	ptsn.drop(['Resp'], axis =1 , inplace = True)
	ptsn_group = pd.DataFrame()
	file_g = pd.DataFrame()
	for q in range(12):
		i = 400 * q
		j = 400 * (q+1)
		file = ptsn[i:j]
		file_g = file.groupby(file['ID']).mean()
		ptsn_group = pd.concat([ptsn_group, file_g], axis = 0)

	pwsn.drop(['Resp'], axis =1 , inplace = True)
	pwsn_group = pd.DataFrame()
	file_g = pd.DataFrame()
	for q in range(12):
		i = 400 * q
		j = 400 * (q+1)
		file = pwsn[i:j]
		file_g = file.groupby(file['ID']).mean()
		pwsn_group = pd.concat([pwsn_group, file_g], axis = 0)

	psatn.drop(['Resp'], axis =1 , inplace = True)
	psatn_group = pd.DataFrame()
	file_g = pd.DataFrame()
	for q in range(12):
		i = 400 * q
		j = 400 * (q+1)
		file = psatn[i:j]
		file_g = file.groupby(file['ID']).mean()
		psatn_group = pd.concat([psatn_group, file_g], axis = 0)


#after rollup need to form quarterly files'''
	partner_data = pd.DataFrame()
	partner_data_info = pd.DataFrame()
	partner_data_scores = pd.DataFrame()
	pinfo.reset_index(drop=True, inplace=True)
	mrp.drop(['ID'], axis=1)
	pwa.drop(['ID'], axis=1)
	mrp.reset_index(drop=True, inplace=True)
	pwa.reset_index(drop=True, inplace=True)
	partner_data_info = pd.concat([pinfo, mrp, pwa], axis=1)
	partner_data_info.reset_index(drop=True, inplace=True)
	psatn_group.reset_index(drop=True, inplace=True)
	pwsn_group.reset_index(drop=True, inplace=True)
	ptsn_group.reset_index(drop=True, inplace=True)
	partner_data_scores = pd.concat([psatn_group,pwsn_group,ptsn_group], axis=1)
	partner_data_scores.reset_index(drop=True, inplace=True)
	partner_data = pd.concat([partner_data_info, partner_data_scores], axis=1)
	partner_data.reset_index(drop=True, inplace=True)

#write database for nn processing and time series processing

	f = open('partner_data.pkl', 'wb')
	pkl.dump(partner_data,f,-1)
	f.close()
