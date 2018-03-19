import pandas as pd
import numpy as np
import pickle as pkl
from bokeh.layouts import layout
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Select,DataTable,TableColumn,Slider
from bokeh.plotting import figure
import pickle as pkl

with open('Time_Series_Predict.pkl', 'rb') as f:
	data = pkl.load(f)
#create data set for slope by id figure
df = data[data.Date == '2014-11-30'].reset_index(drop=True)
df = df.sort_values(by=['Slope'])
drops = df.columns[1:7]
drops.append(df.columns[-1])
df.drop(drops,axis=1)
#set source original and source to be moved with slider
source_original = ColumnDataSource.from_df(df)
source = ColumnDataSource.from_df(df)
#basic figure
f = figure(x='ID',y='Slope',source=source)

curdoc.add_root(f)







metrics_up = pd.read_csv('metrics_up.csv')
features_up = pd.read_csv('features_up.csv')
dataup = pd.read_csv('dataup.csv')
pup = pd.read_csv('partners_up.csv')
partners_up = list(pup['0'])
partners_up=map(str,partners_up)

'''The next builds a data table to be dislayed so that the user can select the partner to be more carefully investigated.  This table is a static table not part of the callback routine.'''
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
