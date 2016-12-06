	
import numpy as np
import pandas as pd

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
	
def remove_chars(df, column):
	#remove whitespaces alphabet and punctation from strings in columns
	df.column = df.column.str.strip().str.replace(' ', '_')	
	return df.column 
	
def fit_dict(df):
	#create dcitionaries for factor matrix operation
	dtrain = []
	dpred=[]
	y=[]
	dfm = pd.melt(df,id_vars=['IDX'])
	null_vec = pd.isnull(dfm.value)
	not_null_vec = pd.notnull(dfm.value)
	df_data = dfm[not_null_vec]
	df_predict = dfm[null_vec]
	for i in range(len(df_data)):
		dtrain.append({'response_id':str(df_data.iloc[i,0]),'ques_id':str(df_data.iloc[i,1])})
		y.append(float(df_data.iloc[i,2]))
	for i in range(len(df_predict)):
		dpred.append({'response_id': str(df_predict.iloc[i,0]),'ques_id': str(df_predict.iloc[i,1])})	
	return dtrain,y,dpred

def vectorize_fit(Xtrain,Xtest,Xpred):
	v = DictVectorizer()
	X_train = v.fit_transform(Xtrain)
	X_test = v.transform(Xtest)
	X_pred = v.transform(Xpred)
	return X_train,X_test,X_pred
	
def fit_fm(X):
	fm = plf.FM(num_factors=10, num_iter=100, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")
	