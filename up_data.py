'''from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, LassoSelectTool
from bokeh.models.widgets import Select,DataTable,TableColumn
from bokeh.plotting import figure'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from bokeh_data import load_data,pick_partners,format_partners,find_importance
'''from jinja2 import Template
from bokeh.util.browser import view
from bokeh.document import Document
from bokeh.embed import autoload_server
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from tornado.ioloop import IOLoop
from flask import Flask, render_template'''

'''flask_app = Flask(__name__)'''
'''utilities'''
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


if __name__ == '__main__':
	'''load and construct column data sources'''
	df = load_data('Partners_Mon_Revenue.csv')
	dfup,dfdown=pick_partners(df,n=20)
	dataup,partners_up,metrics_up=format_partners(dfup)
	datadown,partners_down,metrics_down=format_partners(dfdown)
	'''build feature importances'''
	dfp=load_data('Partner_Data_Complete.csv')
	dfw=load_data('Partner_Features_In_Order.csv')
	features_up=find_importance(dfp,dfw,dfup)
	features_down=find_importance(dfp,dfw,dfdown)
	'''convert column names to strings and months to integers'''
	cols=dataup.columns
	cols=map(str,cols)
	dataup.columns = cols
	mon=dataup.Month
	mon=map(int,mon)
	dataup['Month']=mon
	cols=datadown.columns
	cols=map(str,cols)
	datadown.columns = cols
	mon=datadown.Month
	mon=map(int,mon)
	datadown['Month']=mon
	cols=features_up.columns
	cols=map(str,cols)
	features_up.columns = cols
	metrics_up.to_csv('metrics_up.csv', index = False)
	partners_up=pd.DataFrame(partners_up)
	partners_up.to_csv('partners_up.csv', index=False)
	dataup.to_csv('dataup.csv', index = False)
	features_up.to_csv('features_up.csv', index = False)
