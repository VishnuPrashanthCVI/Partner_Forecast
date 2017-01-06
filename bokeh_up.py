from bokeh.io import curdoc, output_file, show
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, LassoSelectTool 
from bokeh.models.widgets import Select,DataTable,TableColumn
from bokeh.plotting import figure
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from bokeh_data import load_data,pick_partners,format_partners,find_importance
from bokeh.models import BoxSelectTool, LassoSelectTool
#plot.select(BoxSelectTool).select_every_mousemove = False
#plot.select(LassoSelectTool).select_every_mousemove = False

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

'''The next builds a data table to be dislayed so that the viewer can select the partner to be more carefully investigated - this will run from the terminal command as well a run bokeh_up.py in ipython.  After creating the html code just send the html code for review from other parties'''
tablesource = ColumnDataSource()
tablesource.data=ColumnDataSource.from_df(metrics_up)
columns = [TableColumn(field="IDX", title="Partner ID", width=100), TableColumn(field="max", title="Max Rev $M", width=100), TableColumn(field="mean", title="Mean Rev $M", width = 100), TableColumn(field="min", title="Min Rev $M",width = 100), TableColumn(field="predict", title="Predicted Rev $M",width = 150)]
metric = DataTable(source=tablesource, columns=columns, width=800, row_headers=True)

'''The next builds a time series chart to observe selected series.  Hovertool for more detail and dropdowns to select time series to view.'''

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

#build drop down box
IDX = Select(value=partners_up[0], options=partners_up, title='Select Partner')

def IDX_update(attribute, old, new):
	partner = IDX.value
	data = time_data(partner,dataup)
	timesource.data = ColumnDataSource.from_df(data)
	fdata=feature_data(partner,features_up)
	featuresource.data=ColumnDataSource.from_df(fdata)

IDX.on_change("value", IDX_update)

output_file=('bokeh_up.html')
layout = column(metric,IDX,p1,featurelist)
curdoc().add_root(layout)
curdoc().title = "Top Twenty Partners"
'''bokeh serve --show bokeh_up.py 
enter in terminal opened in directory containing python code'''