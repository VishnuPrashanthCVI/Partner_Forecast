def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)for uid in uids[:5]:
   ....:     a = dfp.loc[uid,'Revenue']

import random as r

df = ga.load_data(62)

df['FY'] = [df.iloc[i,1].lstrip('FY') for i in range(df.shape[0])]
df['FY'] = df['FY'].apply(lambda x:int(x))

dfr['FY'] = [dfr.iloc[i,1]].lstrip('FY') for i in range(dfr.shape[0])
dfr['FY'] = dfr['FY'].apply(lambda x:int(x))



bins = 5
names = [1,2,3,4]
df['Rank'] = pd.cut(df['Mean_Sat'], bins, labels = names)

df10 = df[df['FY']==10]
df11 = df[df['FY']==11]

df10_1=df10[df10[' Qtr'] == 1]

Growth = [[-1,1],[.5,1.5],[1.0,2.0],[1.5,2.5],[2.0,3.0],[3.0,7.0]]
Sales = [[-10,10],[10,30],[25,50],[40,100],[75,150]]
df['Rank'].fillna(1, inplace = True)

ef strip_cols(df):
	df['FY'] = [df.iloc[i,1].lstrip('FY') for i in range(df.shape[0])]
	df['FY'] = df['FY'].apply(lambda x:int(x))
	
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm as pl

def strip_year_data(df):
	df['FY'] = [df.iloc[i,2].lstrip('FY') for i in range(df.shape[0])]
	df['FY'] = df['FY'].apply(lambda x:int(x))
	df = df.drop('Year', axis = 1)
	cols = ['Year'] + [col for col in df if col != 'Year']
	df = df[cols]
	cols = ['UID'] + [col for col in df if col != 'UID']
	df = df[cols]
	return df
	
year = []
for yr in dfr.Year:
	if yr == 10:
		year.append(1)
	else:
		year.append(2)
dfr.Year = year

df['Mean_Sat'] = [df.iloc[i,13:54].mean() for i in range(df.shape[0])]
		
uids = []
uids = list(dfr.UID)
dfr.set_index('UID', inplace = True)
df.set_index('UID', inplace = True)
[df.loc['u','Revenue'] = dfr.loc['u','Revenue'] for u in uids]


def perf(df):
	sales =[20, 80, 120, 180, 220]
	devs =[5, 20, 30, 45, 70]
	growth=[.01,.015,.02,.03,.05]
	devg = [.002,.002,.0025,.005]
	discount=[.3,.35,.40,.45,.50]
	devd = [.03,.032,.04,.045,.05]
	n = len(df)
	indices = []
	df['Revenue'] = np.empty((n,1))
	df['Discount'] = np.empty((n,1))
	df['Growth'] = np.empty((n,1))
	
	for i in range(n):
		indices.append(i)
	
	for ind in indices:
		s = int(df.loc[ind,'Rank'])-1
		a = round(ra.normalvariate(sales[s],devs[s]))
		df = df.set_value(ind,'Revenue', a)

	
	return df
		
def add_params(df):
	df['Mktg_Proj'] = [df.iloc[i,ra.randint(13,54)] for i in range(df.shape[0])]	
	df['Svc_Proj'] = [df.iloc[i,ra.randint(13,54)] for i in range(df.shape[0])]
	return dfdf['Revenue'] = np.empty((n,1))
df['Discount'] = np.empty((n,1))
df['Growth'] = np.empty((n,1))

for ind in indices:
	s = int(df.loc[ind,'Rank'])-1	
	b = round(ra.normalvariate(discount[s],devd[s]),2)
	df = df.set_value(ind,'Discount', b)

for ind in indices:	
	s = int(df.loc[ind,'Rank'])-1	
	c = round(ra.normalvariate(growth[s],devg[s]),3)
	df = df.set_value(ind,'Growth', c)		
a

for i in range(len(dfr)):
    a = dfr.iloc[i,0]
    uids.append(a)
X
for i in range(len(df)):
    x = df.iloc[i,-6]
    y = x*20+5
    df.set_value(i,-3,y)

for i in range(len(df11)):
	x = df11.ix[i, 'Mean_Sat']
	if x < 3.5:
		y = 18*x
	else:
		y = 5 + 21*x
		a = round(ra.normalvariate(y,2),2)
	df12.set_value(i, 'Revenue', y)
	
	 

df_one = df_one.sort_values(by = ['Qtr', 'NID'], axis = 0, ascending = [1,1])
df_two = df_two.sort_values(by = ['Qtr', 'NID'], axis = 0, ascending = [1,1])
def set_revenue(df,s,t):
	import random as ra
	for i in range(len(df)):
		x = df.Mean_Sat.iloc[i]
		if x < 3:
			y = s*x
		else:
			y = 15 + t*x
		a = round(ra.normalvariate(y,1),2)
		df.set_value(i, 'Revenue', a)
	return df
	
def rank(df):
	bins = [1,2,3,4,5]
	group_names = [1,2,3,4]
	df['Rank'] = pd.cut(df['Satisfaction'], bins, labels = group_names)
	return df	
	
	#(X_train,y_train,X_test,y_test)=train_test_split(dtrain,ytrain,stratify=y)
	#return X_train, y_train, X_test, y_test, dpred
	
	X1 = df_predict.IDX.values
	X2 = df_predict.variable.values
	X3 = y_pred
	Y1 = df_data.IDX.values
	Y2 = df_data.variable.values
	Y3 = y
	Z1 = np.hstack((X1,Y1))
	Z2 = np.hstack((X2,Y2))
	Z3 = np.hstack((X3,Y3))
	df = pd.DataFrame()
	df['IDX'] = Z1
	df['variable'] = Z2
	df['value'] = Z3
	df.sort_values(by='IDX', inplace = True)
	return df
	
length = 963
dfq11 = dfdata[dfdata.Qtr==1][:length]
dfq12 = dfdata[dfdata.Qtr==2][:length]
dfq13 = dfdata[dfdata.Qtr==3][:length]
dfq14 = dfdata[dfdata.Qtr==4][:length]
dfq2 = dfdata[dfdata.Qtr==5][:length]
dfq11.IDX=range(1,964)
dfq12.IDX=range(1,964)
dfq13.IDX=range(1,964)
dfq14.IDX=range(1,964)
dfq2.IDX=range(1,964)
df11.drop('IDX',axis=1,inplace=True)
df12.drop('IDX',axis=1,inplace=True) 
df13.drop('IDX',axis=1,inplace=True)
df14.drop('IDX',axis=1,inplace=True)
df2.drop('IDX',axis=1,inplace=True)
df11.reset_index(drop=True,inplace=True)
dfq11.reset_index(drop=True,inplace=True)
df12.reset_index(drop=True,inplace=True)
dfq12.reset_index(drop=True,inplace=True)
df13.reset_index(drop=True,inplace=True)
dfq13.reset_index(drop=True,inplace=True)
df14.reset_index(drop=True,inplace=True)
dfq14.reset_index(drop=True,inplace=True)
df2.reset_index(drop=True,inplace=True)
dfq2.reset_index(drop=True,inplace=True)
df11p=pd.concat([dfq11,df11],axis=1)
df12p=pd.concat([dfq12,df12],axis=1)
df13p=pd.concat([dfq13,df13],axis=1)
df14p=pd.concat([dfq14,df14],axis=1)
df2p=pd.concat([dfq2,df2],axis=1)
dfp = pd.concat([df12p,df13p,df14p,df2p])
dfp = dfp.fillna(method='ffill')
dfdown=dfp[dfp.Move_Up != 1]
dfup=dfp[dfp.Move_Down != 1]

import Gazelle as ga
import random as r

df = ga.load_data(62)

df['FY'] = [df.iloc[i,1].lstrip('FY') for i in range(df.shape[0])]
df['FY'] = df['FY'].apply(lambda x:int(x))

dfr['FY'] = [dfr.iloc[i,1]].lstrip('FY') for i in range(dfr.shape[0])
dfr['FY'] = dfr['FY'].apply(lambda x:int(x))



bins = [1,2,3,4,5]
group_names = [1,2,3,4]
df['Rank'] = pd.cut(df[' Overall Satisfaction'], bins, labels = group_names)

df10 = df[df['FY']==10]
df11 = df[df['FY']==11]

df10_1=df10[df10[' Qtr'] == 1]

Growth = [[-1,1],[.5,1.5],[1.0,2.0],[1.5,2.5],[2.0,3.0],[3.0,7.0]]
Sales = [[-10,10],[10,30],[25,50],[40,100],[75,150]]
df['Rank'].fillna(1, inplace = True)

ef strip_cols(df):
	df['FY'] = [df.iloc[i,1].lstrip('FY') for i in range(df.shape[0])]
	df['FY'] = df['FY'].apply(lambda x:int(x))
	
from sklearn.feature_extraction import DictVectorizer
from pyfm import pylibfm as pl

def strip_year_data(df):
	df['FY'] = [df.iloc[i,2].lstrip('FY') for i in range(df.shape[0])]
	df['FY'] = df['FY'].apply(lambda x:int(x))
	df = df.drop('Year', axis = 1)
	cols = ['Year'] + [col for col in df if col != 'Year']
	df = df[cols]
	cols = ['UID'] + [col for col in df if col != 'UID']
	df = df[cols]
	return df
	
year = []
for yr in df.FY:
	if yr == 10:
		year.append(1)
	else:
		year.append(2)
df.Year = year

df['Mean_Sat'] = [df.iloc[i,13:54].mean() for i in range(df.shape[0])]
		
uids = []
uids = list(dfr.UID)
dfr.set_index('UID')
df.set_index('UID')
uids = []
uids = list(dfr.UID)



df.loc[uids[1],'Revenue'] = dfr.loc[uids[1],'Revenue']
	np.any(np.isnan(mat))
	np.all(np.isfinite(mat))
	
def load_data(data):
	na_list = ["Don't Know", "No Experience", "Not Applicable"]
	df = pd.read_csv(data, na_values = na_list)
	return df

def sort_annual_data(df, year1, year2):
	#subset raw data by year
	df_one = df[df['Year'] == str(year1)]
	df_two = df[df['Year'] == str(year2)]
	return df_one, df_two
	
def anon_cols(df):	
	#substitute anonymous string for data columns
	df_dict ={} 
	#Make dictionary of original columns names by row
	for x in range(df.shape[1]):
		df_dict[x] = df.columns[x]
	#Rename columns to anonymous number equal to df_dict keys
	anon = []
	for i in range(1, 1+df.shape[1], 1):
		a = str(i)
		anon.append(a)
	df.columns = anon	
	return df, df_dict

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
	
def set_idx(df):
	#set unique sequential id number 
	for i in range(len(df)):
		df = df.set_value(i,'IDX',str(i))
	return df
	
	
def fit_train_dct(df):
	#create dcitionaries for factor matrix operation
	Xtrain = []
	y=[]
	dfm = pd.melt(df,id_vars=['IDX'])
	not_null_vec = pd.notnull(dfm.value)
	df_data = dfm[not_null_vec]
	for i in range(len(df_data)):
		Xtrain.append({'response_id':str(df_data.iloc[i,0]),'ques_id':str(df_data.iloc[i,1])})
		y.append(float(df_data.iloc[i,2]))
	return Xtrain, y, df_data
	
def fit_pred_dct(df):
	Xpred=[] 
	dfm = pd.melt(df,id_vars=['IDX'])
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
	
def merge_data(df_predict, y_pred, df_data):
	df_predict.drop('value',axis=1,inplace=True)
	df_predict['value']=y_pred
	dfp = pd.concat([df_data,df_predict])
	dfdata = dfp.pivot(index='IDX',columns='variable',values='value')
	dfdata.reset_index(inplace=True)
	return dfdata

def data_prep(dfdata, df11, df12, df13, df14, df2, length=963):
	dfq11 = dfdata[dfdata.Qtr==1][:length]
	dfq12 = dfdata[dfdata.Qtr==2][:length]
	dfq13 = dfdata[dfdata.Qtr==3][:length]
	dfq14 = dfdata[dfdata.Qtr==4][:length]
	dfq2 = dfdata[dfdata.Qtr==5][:length]
	dfq11.IDX=range(1,964)
	dfq12.IDX=range(1,964)
	dfq13.IDX=range(1,964)
	dfq14.IDX=range(1,964)
	dfq2.IDX=range(1,964)
	cols = ['IDX','Move_Up','Move_Down']
	df11.drop(cols,axis=1,inplace=True)
	df12.drop(cols,axis=1,inplace=True) 
	df13.drop(cols,axis=1,inplace=True)
	df14.drop(cols,axis=1,inplace=True)
	df2.drop(cols,axis=1,inplace=True)
	df11.reset_index(drop=True,inplace=True)
	dfq11.reset_index(drop=True,inplace=True)
	df12.reset_index(drop=True,inplace=True)
	dfq12.reset_index(drop=True,inplace=True)
	df13.reset_index(drop=True,inplace=True)
	dfq13.reset_index(drop=True,inplace=True)
	df14.reset_index(drop=True,inplace=True)
	dfq14.reset_index(drop=True,inplace=True)
	df2.reset_index(drop=True,inplace=True)
	dfq2.reset_index(drop=True,inplace=True)
	df11p=pd.concat([dfq11,df11],axis=1)
	df12p=pd.concat([dfq12,df12],axis=1)
	df13p=pd.concat([dfq13,df13],axis=1)
	df14p=pd.concat([dfq14,df14],axis=1)
	df2p=pd.concat([dfq2,df2],axis=1)
	dfp = pd.concat([df12p,df13p,df14p,df2p])
	return dfp,dfq2

def random_forest_up(dfup,dfpred):
	y = np.array(dfup.Move_Up)
	X = dfup.drop(['IDX','Move_Down','Move_Up','Revenue'],axis=1).as_matrix()
	X = StandardScaler().fit_transform(X)
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=71, stratify=y)
	rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=71, class_weight = 'balanced')
	rfc.fit(x_train, y_train)
	rfc_result = rfc.predict(x_test)
	rfc_mse = mean_squared_error(rfc_result, y_test)
	cmup = confusion_matrix(rfc_result, y_test)
	Xpred = StandardScaler().fit_transform(dfpred)
	rfc_predict = rfc.predict(Xpred)
	features = rfc.feature_importances()
	#rfc_predict_up=
	
	
def random_forest_up(dfup,df2):
	y = np.array(dfup.Move_Up)
	X = dfup.drop(['IDX','Move_Down','Move_Up'],axis=1).as_matrix()
	X = StandardScaler().fit_transform(X)
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1, stratify=y)
	rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=71, class_weight = 'balanced')
	rfc.fit(x_train, y_train)
	rfc_result = rfc.predict(x_test)
	rfc_mse = mean_squared_error(rfc_result, y_test)
	prediction = rfc.predict
	
	"df11 = load_data('df11.csv')
	df12 = load_data('df12.csv')
	df13 = load_data('df13.csv')
	df14 = load_data('df14.csv')
	df2 = load_data('df2.csv')
	#prepare data for decision tree on categoricals
	dftt, dft2p =dtdata_prep(dft,df12,df13,df14,df2)
	#run decision tree on categorical data - dtrpred is predicted revenue
	dtrpred,dtrpred_mse,dtrpred_score=dec_tree(dftt,dft2p)
	#prepare data for random forest 
	dfp,df2p = data_prep(dfdata,df12,df13,df14,df2)"
	#X=[];y=[];xtrain=[];xtest=[];ytrain=[];ytest=[]
	#X,y,xtrain,xtest,ytrain,ytest=split_add_data(dfp,dtrpred)
	#merge quarterly revenue data into new fitted data dfdata to be called dfp
	#dfq2 file is x matrix for last quarter predictors in random forest
	#dfp,dfq2 = data_prep(dfdata,df11,df12,df13,df14,df2,length=963)'