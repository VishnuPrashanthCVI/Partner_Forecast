import pandas as pd
import numpy as np

def load_data(data):
	na_list = ["Don't Know", "No Experience", "Not Applicable"]
	df = pd.read_csv(data, na_values = na_list)
	return df
	
def pick_partners(df, n=10):
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


df = load_data('Partners_Mon_Revenue.csv')
dfup,dfdown=pick_partners(df,n=10)
dataup,partners_up,metrics_up=format_partners(dfup)
datadown,partners_down,metrics_down=format_partners(dfdown)
