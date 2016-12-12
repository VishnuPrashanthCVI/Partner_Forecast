
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
import xlsxwriter



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
	dfq11 = dfdata[dfdata.Qtr==1].reset_index(drop=True)[:1926]
	dfq12 = dfdata[dfdata.Qtr==2].reset_index(drop=True)[:1926]
	dfq13 = dfdata[dfdata.Qtr==3].reset_index(drop=True)[:1926]
	dfq14 = dfdata[dfdata.Qtr==4].reset_index(drop=True)[:1926]
	dfq2 = dfdata[dfdata.Qtr==5].reset_index(drop=True)[:1926]
	df12 = df12[:1926]
	df13 = df13[:1926]
	df14 = df14[:1926]
	df2 = df2[:1926]
	df11p=pd.concat([dfq11,df12],axis=1)
	df12p=pd.concat([dfq12,df13],axis=1)
	df13p=pd.concat([dfq13,df14],axis=1)
	df14p=pd.concat([dfq14,df2],axis=1)
	dfp = pd.concat([df11p,df12p,df13p,df14p],ignore_index=True)
	return dfp,dfq2
	
def dtdata_prep(dft, df12, df13, df14, df2):
	#requires dft = partner categorical data and the revenue files
	dftq11 = dft[dft.Qtr==1].reset_index(drop=True)[:1926]
	dftq12 = dft[dft.Qtr==2].reset_index(drop=True)[:1926]
	dftq13 = dft[dft.Qtr==3].reset_index(drop=True)[:1926]
	dftq14 = dft[dft.Qtr==4].reset_index(drop=True)[:1926]
	dftq2 = dft[dft.Qtr==5].reset_index(drop=True)[:1926]
	df12 = df12[:1926]
	df13 = df13[:1926]
	df14 = df14[:1926]
	df2 = df2[:1926]	
	df11p=pd.concat([dftq11,df12],axis=1)
	df12p=pd.concat([dftq12,df13],axis=1)
	df13p=pd.concat([dftq13,df14],axis=1)
	df14p=pd.concat([dftq14,df2],axis=1)
	dftt = pd.concat([df11p,df12p,df13p,df14p],ignore_index=True)
	dftp = pd.concat([dftq11,dftq12,dftq13,dftq14,dftq2],ignore_index=True)
	return dftt, dftp

def dec_tree(dftt,dftp):
	#create X and y arrays for decision tree
	X=pd.DataFrame([])
	y=pd.DataFrame([])
	Z=pd.DataFrame([])
	#fillna just to make sure no NAN to foul up algorithm
	y = dftt.pop('Revenue').fillna(method = 'ffill').as_matrix()
	X = dftt.drop(['Qtr','Country','Customer_Size'], axis = 1).fillna(method = 'ffill')
	Z = pd.get_dummies(X).as_matrix()
	Z = StandardScaler().fit_transform(Z)
	x_train, x_test, y_train, y_test = train_test_split(Z, y, test_size=.2, random_state=71)
	dtr = DecisionTreeRegressor(random_state=71, max_depth=5, max_features=5)
	dtr.fit(x_train, y_train)
	dtr_result=dtr.predict(x_test)
	dtr_mse = mean_squared_error(dtr_result, y_test)
	P = dftp.drop(['Qtr','Country','Customer_Size'], axis = 1).fillna(method='ffill')
	T = pd.get_dummies(P).as_matrix()
	T = StandardScaler().fit_transform(T)
	dtr.fit(Z,y)
	dtrpred = dtr.predict(T)
	return dtrpred,dtr_mse

def split_add_data(dfp,dfq2,dtrpred):
	X = pd.DataFrame([])
	y = pd.DataFrame([])
	y = dfp.pop('Revenue').fillna(method = 'ffill').as_matrix()
	#fill any na's left no matter source as data for random forest
	#add predictions from decision tree
	p = len(dfp)
	dfp['Prediction'] = dtrpred[:p]
	dfq2['Prediction'] = dtrpred[p:]
	X=dfp.drop(['Qtr','Mean_Sat'],axis=1)
	#fill any na's left no matter source as data for random forest
	X.fillna(method = 'ffill', inplace=True)
	X.as_matrix()
	X = StandardScaler().fit_transform(X)
	xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=.2, random_state=71)
	return X,y,xtrain,xtest,ytrain,ytest,dfp,dfq2
	
def random_forest(xtrain,xtest,ytrain,ytest,dfq2):	
	rfc = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=71, verbose=True)
	rfc.fit(xtrain, ytrain)
	rfc_result = rfc.predict(xtest)
	rfc_mse = mean_squared_error(rfc_result, ytest)
	dfq = dfq2.drop(['Qtr','Mean_Sat'], axis = 1)
	Xpred = StandardScaler().fit_transform(dfq)
	rfc_predict = rfc.predict(Xpred)
	rfc_features = rfc.feature_importances_
	rfc_score_train = rfc.score(xtrain,ytrain)
	rfc_score_test = rfc.score(xtest,ytest)
	return rfc_mse,rfc_predict,rfc_features,rfc_score_train,rfc_score_test

def ada_boost(xtrain,xtest,ytrain,ytest,dfq2):
	abc = AdaBoostRegressor(DecisionTreeRegressor(random_state=71, max_depth=5, max_features=5))
	abc.fit(xtrain,ytrain)
	abc_result = abc.predict(xtest)
	abc_mse = mean_squared_error(abc_result, ytest)
	dfq = dfq2.drop('Qtr', axis = 1)
	Xpred = StandardScaler().fit_transform(dfq)
	abc_predict = abc.predict(Xpred)
	abc_features = abc.feature_importances_
	abc_score = abc.score(xtrain,ytrain)
	return abc_mse,abc_predict,abc_features,abc_score

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
	dftt, dftp =dtdata_prep(dft,df12,df13,df14,df2)
	#run decision tree on categorical data - dtrpred is predicted revenue
	dtrpred,dtr_mse = dec_tree(dftt,dftp)
	#prepare data to merge and for random forest
	dfp, dfq2 = data_prep(dfdata, df12, df13, df14, df2)
	#merge data for random forest
	X,y,xtrain,xtest,ytrain,ytest,dfp,dfq2=split_add_data(dfp,dfq2,dtrpred)
	#run random forest
	rfc_mse,rfc_predict,rfc_features,rfc_score_train,rfc_score_test = random_forest(xtrain,xtest,ytrain,ytest,dfq2)
	rfcmetrics=[rfc_mse,rfc_score_train,rfc_score_test]
	rfcmetrics=[rfc_mse,rfc_score_train,rfc_score_test]
	pd.DataFrame(rfcmetrics).to_csv('rfcmetrics.csv',index=False)
	#run adaboost regressor
	#abc_mse,abc_predict,abc_features,abc_score=ada_boost(xtrain,xtest,ytrain,ytest,dfq2)
		#plot features by weight
	features = pd.DataFrame(rfc_features)
	features['Feature']=dfq2.drop(['Qtr','Mean_Sat'], axis = 1).columns
	cols = ['Weight','Feature']
	features.columns=cols
	features.sort_values(by='Weight',inplace=True,ascending=False)
	features.to_csv('features.csv',index=False)
	#plot features
	Y = np.arange(10)
	X = list(features.Weight[:10])
	f = list(features.Feature[:10])
	width = .5
	fig = plt.figure(figsize=(8,6))
	plt.barh(Y,X,width,color='b')
	plt.ylabel('Feature',fontsize='large')
	plt.xlabel('Weight',fontsize='large')
	plt.yticks(Y+width/2.0,f,fontsize='medium')
	plt.subplots_adjust(left=0.5)
	plt.xlim(0,.2)
	plt.title('Top Ten Features By Weight')
	fig.savefig('features.png', dpi=fig.dpi)
	plt.show()
	plt.close()
	#find performance by class = up, down, same
	df3=pd.DataFrame(rfc_predict)
	df3.columns=['Revenue']
	t=.1
	df2=df2[:len(df3)]
	df3['IDX']=range(len(df3))
	dfu=df3[df3.Revenue>=(1+t)*df2.Revenue]
	dfd=df3[df3.Revenue<=(1-t)*df2.Revenue]
	players=(dfu.shape[0],df3.shape[0]-dfd.shape[0]-dfu.shape[0],dfd.shape[0])
	players_rev=(dfu.Revenue.sum(),df3.Revenue.sum()-dfu.Revenue.sum()-dfd.Revenue.sum(),dfd.Revenue.sum())
	performance=pd.DataFrame()
	performance['Rank']=['Up','Same','Down']
	performance['Player']=players
	performance['Revenue']=players_rev
	performance.to_csv('Dealer Performance.csv',index=False)
	#plot revenue and players
	Y = np.arange(3)
	X = list(performance.Player)
	f = list(performance.Rank)
	width = .5
	fig = plt.figure(figsize=(6,4))
	plt.barh(Y,X,width,color='b')
	plt.ylabel('Rank',fontsize='large')
	plt.xlabel('Count',fontsize='large')
	plt.yticks(Y+width/2.0,f,fontsize='large')
	plt.subplots_adjust(left=0.15,bottom=.2)
	plt.title('Number of Partners By Rank')
	fig.savefig('Partners.png', dpi=fig.dpi)
	plt.show()#plot revenue and players
	plt.close()
	#compute sales forecasts and display
	Y = np.arange(3)
	X = list(performance.Revenue/1000.)
	f = list(performance.Rank)
	width = .5
	fig = plt.figure(figsize=(6,4))
	plt.barh(Y,X,width,color='b')
	plt.ylabel('Rank',fontsize='large')
	plt.xlabel('Revenue In Millions',fontsize='large')
	plt.yticks(Y+width/2.0,f,fontsize='large')
	plt.subplots_adjust(left=0.2,bottom=.2)
	plt.title('Partners Revenue By Rank')
	fig.savefig('Partners_Revenue.png', dpi=fig.dpi)
	plt.show()
	plt.close()
	#plot sales history and forecast	
	


	
	
	
	
	






