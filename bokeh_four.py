import pandas as pd
import numpy as np
import pickle as pkl
from os.path import dirname, join
from bokeh.layouts import layout,column,row,widgetbox
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource,CustomJS,Range1d,PanTool,HoverTool,ResetTool
from bokeh.models.widgets import Select,DataTable,TableColumn,Slider
from bokeh.plotting import figure

with open('Partner_Select_Mon.pkl', 'rb') as f:
	data = pkl.load(f)

'''The next builds a data table to be dislayed so that the user can select the partner to be more carefully investigated.  This table is a static table not part of the callback routine.'''
date = data.Date.max()
df = data[data.Date == date]
cols = data.columns[1:4]
df = df.drop(cols,axis=1)
cols = df.columns[1:]
for col in cols:
	df[[col]] = df[col].apply(lambda x: pd.Series.round(pd.Series(x),2))
tablesource = ColumnDataSource()
tablesource.data=ColumnDataSource.from_df(df)
columns = [TableColumn(field='ID', title="Partner ID", width=100), TableColumn(field='Pred', title="Predicted", width=125), TableColumn(field='Low', title="Low Predicted", width = 125), TableColumn(field='High', title="High Predicted", width = 125), TableColumn(field='Trend', title="Baseline", width = 125),	TableColumn(field='Change', title="Predicted Change", width = 125)]
partners = DataTable(source=tablesource, columns=columns, width=700, row_headers=False)

'''The next builds a time series chart to observe selected series.  Select box changes id to present on update'''
#select data by id and source update function
def time_data(id, data):
	dt = data[data.ID == id]
	actual = dt.Act[:-3]
	dt = dt.drop('Act',axis=1)
	dt['Act'] = actual
	return dt
#update data
dt = time_data(data.ID[0], data)
timesource = ColumnDataSource()
timesource.data = ColumnDataSource.from_df(dt)

# Create the basic time plot
plot = figure(x_axis_type='datetime')
plot.line(x='Date',y='Act',source=timesource,line_color='black',line_dash='solid',line_width=1,legend='Actual')
plot.line(x='Date',y='Pred',source=timesource,line_color='blue',line_dash='dashed',line_width=1,legend='Predicted')
plot.line(x='Date',y='High',source=timesource,line_color='red',line_dash='dashed',line_width=1,legend='High')
plot.line(x='Date',y='Low',source=timesource,line_color='red',line_dash='dashed',line_width=1,legend='Low')
plot.circle(x='Date',y='Trend',source=timesource,line_color='green',line_dash='dashed',line_width=1,legend='Trend')

plot.plot_width=1200
plot.plot_height=750
plot.background_fill_color="lightgreen"
plot.background_fill_alpha=0.3

plot.title.text="Projected Order Input"
plot.title.text_color="black"
plot.title.text_font="Arial"
plot.title.text_font_size="28px"
plot.title.align="center"

plot.xaxis.axis_label="Date"
plot.yaxis.axis_label="Order Input"
plot.axis.axis_label_text_font="Arial"
plot.axis.axis_label_text_font_size="18px"

plot.border_fill_color='whitesmoke'

plot.tools=[PanTool(),ResetTool()]
hover=HoverTool(tooltips=[('ID','@ID'),('Predicted','@Pred'),('High Prediction','@High'),('Low Prediction','@Low'),('Trend','@Trend')])
plot.add_tools(hover)
plot.toolbar_location = 'above'
plot.toolbar.logo=None

plot.grid
plot.min_border = 100

plot.legend.location='top_left'
plot.legend.background_fill_color = 'green'
plot.legend.background_fill_alpha=.5

'''the following builds the feature datatable that is updated by id'''
filestr = 'Feature_Importance_ID.pkl'
with open(filestr,'rb') as f:
	dd = pkl.load(f)
#constructs 3 dataframes for rank,feature,weight with columns = id
cols = [str(f) for f in dd.keys()]
num_features = len(dd[int(cols[0])])
index = range(num_features)
fdata = pd.DataFrame(index=index,columns=cols)
wdata = pd.DataFrame(index=index,columns=cols)
rdata = pd.DataFrame(index=index,columns=cols)
for c in cols:
	rdata[c] = [dd[int(c)][i][0] for i in index]
	fdata[c] = [dd[int(c)][i][1] for i in index]
	wdata[c] = [round(dd[int(c)][i][2],3) for i in index]
#construct df for feature source
def feature_data(id,index=index,rdata=rdata,fdata=fdata,wdata=wdata):
	ds = pd.DataFrame(index=index,columns=['Feature','Weight','Rank'])
	ds['Feature'] = fdata[id]
	ds['Weight'] = wdata[id]
	ds['Rank'] = rdata[id]
	return ds
#build feature datasource
ds = feature_data(cols[0])
featuresource = ColumnDataSource()
featuresource.data = ColumnDataSource.from_df(ds)
#build datatable
feature_columns = [TableColumn(field='Feature', title="Feature", width=200), TableColumn(field='Weight', title="Weight", width=125), TableColumn(field='Rank', title="Rank", width = 100)]
features = DataTable(source=featuresource,columns=feature_columns,width=425,row_headers=False)

'''build drop down box and callback function'''
#select box
options = [str(x) for x in list(data.ID.unique())]
ID = Select(value=options[0], options=options, title='Select Partner')
#update function
def ID_update(attribute, old, new):
	id = int(ID.value)
	dt = time_data(id,data)
	timesource.data = ColumnDataSource.from_df(dt)
	ds = feature_data(ID.value)
	featuresource.data=ColumnDataSource.from_df(ds)

ID.on_change("value", ID_update)

layout1 = layout(partners,ID,plot,features)
curdoc().add_root(layout1)
curdoc().title = 'Selected Partners'
