import pandas as pd
import numpy as np

def load_data(data):
	na_list = ["Don't Know", "No Experience", "Not Applicable"]
	df = pd.read_csv(data, na_values = na_list)
	return df
	
def select_change_pid(df, n=10):
	dfp=df
	dfp['Change']=(dfp['15']-dfp['12'])/df['12']
	dfp.sort_values(by = 'Change', inplace=True, ascending=False)
	dfup=dfp.head(n).drop('Change',axis=1)
	dfdown=dfp.tail(n).drop('Change',axis=1)
	
	
df = pd.read_csv('Partners_Mon_Revenue.csv')
data = pd.DataFrame()
data['Month']=df.columns[1:]
dfdata=df.iloc[:,1:]
dfdata=dfdata.transpose()
data = data.reset_index(drop=True)
dfdata = dfdata.reset_index(drop=True)
data = pd.concat([data,dfdata],axis=1)