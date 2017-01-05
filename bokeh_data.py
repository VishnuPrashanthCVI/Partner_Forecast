import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

"""Bokeh accepts data in column format e.g., ColumnDataSource.  DataTables also require reshaping to column source.  Also the partner count in this sample is way too large for bokeh to display in dropowns etc.  This data has been filtered for top and bottom performers by setting n in def pick_partners"""

def load_data(data):
	na_list = ["Don't Know", "No Experience", "Not Applicable"]
	df = pd.read_csv(data, na_values = na_list)
	return df
	
def pick_partners(df, n=20):
	dfp=df
	dfp['Change']=(dfp['15']-dfp['12'])/df['12']
	dfp.sort_values(by = 'Change', inplace=True, ascending=False)
	dfup=dfp.head(n).drop('Change',axis=1)
	dfdown=dfp.tail(n).drop('Change',axis=1)
	return dfup,dfdown
	
def format_partners(df):
	data = pd.DataFrame()
	data['Month']=df.columns[1:]
	dfdata=df.iloc[:,1:]
	dfdata=dfdata.transpose()
	data = data.reset_index(drop=True)
	dfdata = dfdata.reset_index(drop=True)
	data = pd.concat([data,dfdata],axis=1)
	data.reset_index(drop=True,inplace=True)
	partners=list(data.columns[1:])
	partners = map(str,partners)
	metrics={}
	datamean=data.iloc[:,1:].mean().round(1)
	datamax=data.iloc[:,1:].max().round(1)
	datamin=data.iloc[:,1:].min().round(1)
	datapredict=data.iloc[data.shape[0]-1,1:]
	metrics={'mean':datamean}
	metrics['max']=datamax
	metrics['min']=datamin
	metrics['predict']=datapredict
	metrics=pd.DataFrame(metrics)
	metrics=metrics.reset_index()
	metrics.columns=['IDX','max','mean','min','predict']
	return data, partners, metrics

def find_importance(dfp,dfw,df,n=20):
	dfdata=pd.DataFrame()
	for i in range(len(df)):
		dfrow=dfp[dfp.index==df.index[i]]
		dfdata=dfdata.append(dfrow)
	dfww=dfw.drop('Feature',axis=1)
	X = np.array(dfww)
	d=dfdata.drop('IDX',axis=1)
	cols=d.columns
	d=StandardScaler().fit_transform(d)
	dd=np.array(d)
	pw=dd*X.T
	pw=pd.DataFrame(pw)
	pw.index=dfdata.index
	pw.columns=cols
	pf=pd.DataFrame()
	for i in range(len(pw)):
		prow=pw.iloc[i,:]
		prow.sort_values(inplace=True,ascending=False)
		pf[prow.name]=list(prow.index[:n])
	return pf
	
	
	
