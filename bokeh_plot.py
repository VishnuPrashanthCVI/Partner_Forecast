from bokeh.io import curdoc, output_file, show
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, LassoSelectTool 
from bokeh.models.widgets import Select,DataTable,TableColumn
from bokeh.plotting import figure
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from bokeh_data import load_data,pick_partners,format_partners,find_importance

'''load and construct column data sources'''
df = load_data('Partners_Mon_Revenue.csv')
dfup,dfdown=pick_partners(df,n=20)
dataup,partners_up,metrics_up=format_partners(dfup)
datadown,partners_down,metrics_down=format_partners(dfdown)
dfp=load_data('Partner_Data_Complete.csv')
dfw=load_data('Partner_Features_In_Order.csv')
features_up=find_importance(dfp,dfw,dfup)
features_down=find_importance(dfp,dfw,dfdown)

source = ColumnDataSource()
source.data=ColumnDataSource.from_df(metrics_up)
columns = [TableColumn(field="IDX", title="Partner ID", width=100), TableColumn(field="max", title="Max Rev", width=100), TableColumn(field="mean", title="Mean Rev", width = 100), TableColumn(field="min", title="Min Rev",width = 100), TableColumn(field="predict", title="Predicted Rev",width = 150)]
metric = DataTable(source=source, columns=columns, width=800)
layout=column(metric)
output_file=('metrics_up.html')
show(layout)
curdoc().add_root(layout)
curdoc().title = "Top Ten Partner Metrics"
show(layout)
#show(table)'''
