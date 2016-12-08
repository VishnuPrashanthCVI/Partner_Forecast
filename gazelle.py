
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm as plf
import random as ra 
import string
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import tree


def load_data(data):
	na_list = ["Don't Know", "No Experience", "Not Applicable"]
	df = pd.read_csv(data, na_values = na_list)
	return df

def factor_data(df,x=1,y=3):
	#remove unwanted columns from factor engine
	y += 1
	for i in range(x, y, 1):
		df.drop(df.columns[i], axis = 1, inplace = True)
	return df
		
def cols_strip(df):
	#remove whitepaces and substitute spaces in names with  underscore
	df.columns = df.columns.str.strip().str.replace(' ', '_')
	return df

def set_revenue(dfq,s,t):
	import random as ra
	#f['Revenue']=range(len(dfq))
	for i in range(len(dfq)):
		x = dfq.Mean_Sat.iloc[i]
		if x < 3:
			y = s*x
		else:
			y = 15 + t*x
		a = round(ra.normalvariate(y,1),2)
		dfq.set_value(i, 'Revenue', a)
	return dfq

def set_idx(df):
	#set unique sequential id number 
	for i in range(len(df)):
		df = df.set_value(i,'IDX',str(i))
	return df
	
def melt_train_dict(dff):
	#create dcitionaries for factor matrix operation
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
	
def melt_pred_dict(dff):
	#create factor program prediction 
	Xpred=[]
	dff.IDX = range(dff.shape[0]) 
	dfm = pd.melt(dff,id_vars=['IDX'])
	null_vec = pd.isnull(dfm.value)
	df_predict = dfm[null_vec]
	for i in range(len(df_predict)):
		Xpred.append({'response_id': str(df_predict.iloc[i,0]),'ques_id': str(df_predict.iloc[i,1])})
	return Xpred, df_predict 
		
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

def data_prep(dfdata, df12, df13, df14, df2):
	#requires dfdata to be the filled data from the recommender
	#the revenue files need to be loaded in main file 
	dfq11 = dfdata[dfdata.Qtr==1].reset_index(drop=True)
	dfq12 = dfdata[dfdata.Qtr==2].reset_index(drop=True)
	dfq13 = dfdata[dfdata.Qtr==3].reset_index(drop=True)
	dfq14 = dfdata[dfdata.Qtr==4].reset_index(drop=True)
	dfq2 = dfdata[dfdata.Qtr==5].reset_index(drop=True)
	df11p=pd.concat([dfq11,df12],axis=1,ignore_index=True)
	df12p=pd.concat([dfq12,df13],axis=1,ignore_index=True)
	df13p=pd.concat([dfq13,df14],axis=1,ignore_index=True)
	df14p=pd.concat([dfq14,df2],axis=1,ignore_index=True)
	dfp = pd.concat([df11p,df12p,df13p,df14p],ignore_index=True)
	return dfp,dfq2
	
def dtdata_prep(dft, df12, df13, df14, df2):
	#requires dft = partner categorical data and the revenue files
	dftq11 = dft[dft.Qtr==1].reset_index(drop=True)
	dftq12 = dft[dft.Qtr==2].reset_index(drop=True)
	dftq13 = dft[dft.Qtr==3].reset_index(drop=True)
	dftq14 = dft[dft.Qtr==4].reset_index(drop=True)
	dftq2 = dft[dft.Qtr==5].reset_index(drop=True)
	df11p=pd.concat([dftq11,df12],axis=1,ignore_index=True)
	df12p=pd.concat([dftq12,df13],axis=1,ignore_index=True)
	df13p=pd.concat([dftq13,df14],axis=1,ignore_index=True)
	df14p=pd.concat([dftq14,df2],axis=1,ignore_index=True)
	dft = pd.concat([df11p,df12p,df13p,df14p],ignore_index=True)
	return dft, dftq2

def dec_tree(dft,dftq2):
	#create X and y arrays for decision tree
	X=pd.DataFrame([])
	y=pd.DataFrame([])
	#fillna just to make sure no NAN to foul up algorithm
	y = dft.pop('Revenue').fillna(method = 'ffill').as_matrix()
	X = dft.drop(['IDX','Qtr','Country'], axis = 1).fillna(method = 'ffill')
	X = pd.get_dummies(X).as_matrix()
	X = StandardScaler().fit_transform(X)
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=71)
	dtr = DecisionTreeRegressor(random_state=71, max_depth=5, max_features=5)
	dtr.fit(x_train, y_train)
	dtr_result=dtr.predict(x_test)
	dtr_mse = mean_squared_error(dtr_result, y_test)
	dtr.fit(X,y)
	dtrpred = dtr.predict(X)
	dtrpred_mse=mean_squared_error(dtrpred,y)
	dtrpred_score = dtr.score(X,y)
	return dtrpred,dtrpred_mse,dtrpred_score

def split_add_data(dfp,dtrpred):
	X = pd.DataFrame([])
	y = pd.DataFrame([])
	y = dfp.pop('Revenue').fillna(method = 'ffill').as_matrix()
	#fill any na's left no matter source as data for random forest
	#add predictions from decision tree
	dfp['Prediction'] = dtrpred
	X=dfp.drop(['IDX','Qtr'],axis=1)
		#fill any na's left no matter source as data for random forest
	X.fillna(method = 'ffill', inplace=True).as_matrix()
	X = StandardScaler().fit_transform(X)
	xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, random_state=71)
	return X,y,xtrain,xtest,ytrain,ytest
	
def random_forest(xtrain,xtest,ytrain,ytest,dfq2):	
	rfc = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=71, class_weight = 'balanced')
	rfc.fit(xtrain, ytrain)
	rfc_result = rfc.predict(xtest)
	rfc_mse = mean_squared_error(rfc_result, ytest)
	rfc_cm = confusion_matrix(rfc_result, ytest)
	Xpred = StandardScaler().fit_transform(dfq2)
	rfc_predict = rfc.predict(Xpred)
	rfc_features = rfc.feature_importances()
	rfc_score = rfc.score(xtrain,ytrain)
	return rfc_mse,rfc_cm,rfc_predict,rfc_features,rfc_score

def ada_boost(xtrain,xtest,ytrain,ytest,dfq2):
	abc = AdaBoostRegressor(DecisionTreeRegressor(class_weight = 'balanced'), learning_rate=0.1, n_estimators=100, random_state=71)
	abc.fit(xtrain,ytrain)
	abc_result = abc.predict(xtest)
	abc_mse = mean_squared_error(abc_result, ytest)
	abc_cm = confusion_matrix(abc_result, ytest)
	Xpred = StandardScaler().fit_transform(dfq2)
	abc_predict = abc.predict(Xpred)
	abc_features = abc.feature_importances()
	abc_score = abc.score(xtrain,ytrain)
	return abc_predict_mse,abc_cm,abc_predict,abc_features,abc_score

def plot_confusion_matrix(cm, classes = ['Positive', 'Negative'],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	    plt.text(j, i, cm[i, j],
	             horizontalalignment="center",
	             color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True Label')
	plt.xlabel('Predicted Label')
	
	
if __name__ == '__main__':
	df = load_data('Partners_Data.csv')
	#restrict to USA and Canada dealers
	df = df[df.Region == 'US & CANADA']
	#drop unwanted category columns for pylibFM
	cols = ['Country', 'RTM', 'Function', 'Customer_Size', 'Customer_Segment','Region','NID', 'Year', 'Certification']
	dtcols = ['Country', 'RTM', 'Function', 'Customer_Size', 'Qtr', 'Customer_Segment', 'Certification']
	#drop the unwanted cols - dfdt is for decision tree and dff for pylibFM
	dft = df[dtcols]
	dff = df.drop(cols, axis=1)
	dft.reset_index(drop=True,inplace=True)
	dff.reset_index(drop=True,inplace=True)
	qtr = dff.pop('Qtr')
	idx = dff.pop('IDX')
	#Create data dictionaries for factorization
	X,y,df_data = melt_train_dict(dff)
	#create training and test data
	Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=71)
	#create prediction sets
	Xpred, df_predict = melt_pred_dict(dff)
	#vectorize data sets to fit fm model
	X_train, X_test, X_pred = vectorize(Xtrain,Xtest,Xpred)
	#fit FM model
	fm = plf.FM(num_factors=5, num_iter=50, verbose=True, task="regression", initial_learning_rate=0.002, learning_rate_schedule="optimal")
	fm.fit(X_train,ytrain)
	#test prediction
	y_test = fm.predict(X_test)
	mse_fm=mean_squared_error(ytest,y_test)
	print("FM MSE: %.4f" % mean_squared_error(ytest,y_test))
	y_pred = fm.predict(X_pred)
	#merge factored data to make new data set dfdata without categorical data
	dfdata= merge_data(df_predict, y_pred, df_data, qtr)
	#load revenue by quarter and IDX data
	df12 = load_data('df12.csv')
	df13 = load_data('df13.csv')
	df14 = load_data('df14.csv')
	df2 = load_data('df2.csv')
	#clear memory locations
	z=pd.DataFrame([])
	x_train=z;x_test=z;y_train=z;y_test=z;X=z;y=z;ytrain=z;Xtrain=z;Xtest=z;ytrain=z;ytest=z;ypred=z;y_pred=z
	#prepare data for decision tree on categoricals
	dftt, dftq2 =dtdata_prep(dft,df12,df13,df14,df2)
	#run decision tree on categorical data - dtrpred is predicted revenue
	#dtrpred,dtrpred_mse,dtrpred_score=dec_tree(dftt,dfq2)
	
	
	
	
	
	
	
	
	
	
	





