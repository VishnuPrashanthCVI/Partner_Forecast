import os as os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import math as ma
from sklearn.ensemble import RandomForestRegressor as RFR

#set seed for reproducibility
seed = 73
np.random.seed(73)
#fetch data
with open('Partner_Select_Qtr.pkl', 'rb') as f:
	data = pkl.load(f)

#drop columns for classifying data and date variable
cols = data.columns[1:9]
cols = cols.append(data.columns[-1:])
df = data.drop(cols,axis=1)
#construct column list for feature importance
cols = df.columns[1:-1]
#create id list in df.ID
idx = set(list(int(i) for i in df.ID))
#iterate through partner id constructing dataframe of feature importances
scores = {}
feature_importance = {}
columns = ['ID']
columns.extend(cols)
#iterate through selected id's
for id in idx:
	dr = df[df['ID'] == id]
	rev = dr.pop('Rev').values.reshape(-1,1)
	dr = dr.drop(['ID'],axis=1)
	#scale data
	scaler = StandardScaler()
	X = scaler.fit_transform(dr)
	y = scaler.fit_transform(rev).ravel()
	rfr = RFR(n_estimators=25,random_state=seed,oob_score=True,n_jobs=-1)
	rfr.fit(X,y)
	fi = rfr.feature_importances_
	rank = np.argsort(fi)[::-1]
	#fs = list(fi.sort())[::-1]
	feature_weight = [fi[r] for r in rank]
	feature_list=[cols[r] for r in rank]
	feature_rank = range(1,len(rank)+1)
	features = list(zip(feature_rank,feature_list,feature_weight))
	feature_importance[id] = features
	scores[id] = rfr.score(X,y)
#pickle and save feature importances by ID
filestr = 'Feature_Importance_ID.pkl'
with open(filestr,'wb') as f:
	pkl.dump(feature_importance,f,-1)
#pickle and save scores by ID dictionary
filestr = 'RFR_Scores_ID.pkl'
with open(filestr,'wb') as f:
	pkl.dump(scores,f,-1)

#compute RFR on entire sample set with no ID selection
cols = data.columns[0:9]
cols = cols.append(data.columns[-1:])
df = data.drop(cols,axis=1)
#feature names
cols = df.columns
#form y target
rev = df.pop('Rev').values.reshape(-1,1)
#scale data
scaler = StandardScaler()
X = scaler.fit_transform(df)
y = scaler.fit_transform(rev).ravel()
#build regressor model
rfr = RFR(n_estimators=25,random_state=seed,oob_score=True,n_jobs=-1)
rfr.fit(X,y)
#set feature importances and rank
fi = rfr.feature_importances_
rank = np.argsort(fi)[::-1]
#create list of feature importance info
feature_rank = range(1,len(rank)+1)
feature_weight = [fi[r] for r in rank]
feature_list=[cols[r] for r in rank]
features = list(zip(feature_rank,feature_list,feature_weight))
score =rfr.score(X,y)

#save results
filestr = 'Feature_Importance.pkl'
with open(filestr,'wb') as f:
	pkl.dump(features,f,-1)
#pickle and save scores by ID dictionary
filestr = 'RFR_Score.pkl'
with open(filestr,'wb') as f:
	pkl.dump(score,f,-1)
