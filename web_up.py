
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, LassoSelectTool 
from bokeh.models.widgets import Select,DataTable,TableColumn
from bokeh.plotting import figure
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from bokeh_data import load_data,pick_partners,format_partners,find_importance
from jinja2 import Template
from bokeh.util.browser import view
from bokeh.document import Document
from bokeh.embed import autoload_server
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from tornado.ioloop import IOLoop
from flask import Flask, render_template

flask_app = Flask(__name__)
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

'''The next builds a data table to be dislayed so that the viewer can select the partner to be more carefully investigated.  This table is a static table not part of the callback routine.'''
tablesource = ColumnDataSource()
tablesource.data=ColumnDataSource.from_df(metrics_up)
columns = [TableColumn(field="IDX", title="Partner ID", width=100), TableColumn(field="max", title="Max Rev $M", width=100), TableColumn(field="mean", title="Mean Rev $M", width = 100), TableColumn(field="min", title="Min Rev $M",width = 100), TableColumn(field="predict", title="Predicted Rev $M",width = 150)]
metric = DataTable(source=tablesource, columns=columns, width=800, row_headers=True)
'''The next function modifies the document for the flask app'''
def modify_doc(doc):
	'''The next builds a time series chart to observe selected series.'''
	def time_data(partner, d):
		data=pd.DataFrame()
		data['M']=d['Month']
		data['R']=d[partner]
		data = data.dropna()
		return data
	data=time_data(partners_up[0], dataup)
	timesource=ColumnDataSource.from_df(data)
	timesource=ColumnDataSource(timesource)
	# Create the basic time plot
	p1 = figure(title="Prior Months Revenue", plot_width=800, plot_height=500,tools="pan, wheel_zoom, box_select, lasso_select, tap, reset")
	p1.line("M","R", source=timesource)
	p1.title.text_font_size = "25px"
	p1.title.align = "center"
	p1.xaxis.axis_label='Prior Months'
	p1.yaxis.axis_label='Monthly Revenue In Thousands'
	'''The next builds a table of most important features for each partner selected'''
	def feature_data(partner, d):
		data=pd.DataFrame()
		data['F']=d[partner]
		data = data.dropna()
		return data

	fdata=feature_data(partners_up[0], features_up)
	featuresource = ColumnDataSource()
	featuresource.data=ColumnDataSource.from_df(fdata)
	columns = [TableColumn(field="F", title="Most Important Features", width=200)]

	featurelist = DataTable(source=featuresource, columns=columns, width=400, row_headers=True)

	'''build drop down box and callback function'''
	IDX = Select(value=partners_up[0], options=partners_up, title='Select Partner')

	def IDX_update(attribute, old, new):
		partner = IDX.value
		data = time_data(partner,dataup)
		timesource.data = ColumnDataSource.from_df(data)
		fdata=feature_data(partner,features_up)
		featuresource.data=ColumnDataSource.from_df(fdata)

	IDX.on_change("value", IDX_update)
	layout = column(metric,IDX,p1,featurelist)
	doc.add_root(layout)

	'''end of modify doc bokeh app'''

bokeh_app = Application(FunctionHandler(modify_doc))
io_loop = IOLoop.current()
server = Server({'/bkapp': bokeh_app}, io_loop=io_loop, allow_websocket_origin=["localhost:8080"])
server.start()


@flask_app.route('/', methods=['GET'])
def bkapp_page():
    script = autoload_server(model=None, url='http://localhost:5006/bkapp')
    return render_template("embed.html", script=script)

if __name__ == '__main__':
    from tornado.httpserver import HTTPServer
    from tornado.wsgi import WSGIContainer
    from bokeh.util.browser import view

    print('Opening Flask app with embedded Bokeh application on http://localhost:8080/')

    # This uses Tornado to server the WSGI app that flask provides. Presumably the IOLoop
    # could also be started in a thread, and Flask could server its own app directly
    http_server = HTTPServer(WSGIContainer(flask_app))
    http_server.listen(8080)

    io_loop.add_callback(view, "http://localhost:8080/")
    io_loop.start()




    
