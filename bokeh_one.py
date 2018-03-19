import pandas as pd
import numpy as np
import pickle as pkl
from os.path import dirname, join
from bokeh.layouts import layout,column,row,widgetbox
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource,PanTool,ResetTool,CustomJS,HoverTool
from bokeh.models.widgets import Select,Button
from bokeh.plotting import figure, gridplot

with open('Time_Series_Predict.pkl', 'rb') as f:
	data = pkl.load(f)

#create slope data set using idx
data_last = data[data.Date == '2014-11-30'].reset_index(drop=True)
data_slope = data_last.sort_values(by=['Slope'],ascending=True)
data_slope['IDX'] = range(1,data_slope.shape[0]+1)
data_change = data_last.sort_values(by=['Change'],ascending=True)
data_change['IDX'] = range(1,data_change.shape[0]+1)
#set source original for slope and change

anames = ['C', 'NE', 'SE', 'SW', 'W', 'NW']
tnames = ['Central', 'NorthEast', 'SouthEast', 'SouthWest', 'West', 'NorthWest']
names = list((zip(anames,tnames)))

ds = data_slope[data_slope.Territory == anames[0]]
dc = data_change[data_change.Territory == anames[0]]

source_s = ColumnDataSource(ds)
source_c = ColumnDataSource(dc)

#basic slope figure
plot = figure()
plot.circle(x='IDX',y='Slope',color='darkgreen',size=5,alpha=.5, source=source_s)

plot.plot_width=650
plot.plot_height=400
plot.background_fill_color="lightgreen"
plot.background_fill_alpha=0.3

plot.title.text="Projected Relative Change"
plot.title.text_color="black"
plot.title.text_font="Arial"
plot.title.text_font_size="28px"
plot.title.align="center"

plot.xaxis.axis_label="Partners"
plot.yaxis.axis_label="Growth Ratio"
plot.axis.axis_label_text_font="Arial"
plot.axis.axis_label_text_font_size="18px"

plot.border_fill_color='whitesmoke'

plot.tools=[PanTool(),ResetTool()]
hover=HoverTool(tooltips=[('ID','@ID'),('Class','@Class'),('Territory','@Territory')])
plot.add_tools(hover)
plot.toolbar_location = 'above'
plot.toolbar.logo=None

plot.grid
plot.min_border_right = 80
plot.min_border_left = 80


#basic figure
plot1 = figure()
plot1.circle(x='IDX',y='Change',color='darkgreen',size=5,alpha=.5,source=source_c)

plot1.plot_width=650
plot1.plot_height=400
plot1.background_fill_color="lightgreen"
plot1.background_fill_alpha=0.3

plot1.title.text="Projected Absolute Change"
plot1.title.text_color="black"
plot1.title.text_font="Arial"
plot1.title.text_font_size="28px"
plot1.title.align="center"

plot1.xaxis.axis_label="Partners"
plot1.yaxis.axis_label="Absolute Change"
plot1.axis.axis_label_text_font="Arial"
plot1.axis.axis_label_text_font_size="18px"

plot1.border_fill_color='whitesmoke'
plot1.min_border_right = 80
plot1.min_border_left = 80

plot1.tools=[PanTool(),ResetTool()]
hover=HoverTool(tooltips=[('ID','@ID'),('Class','@Class'),('Territory','@Territory')])
plot1.add_tools(hover)
plot1.toolbar_location = 'above'
plot1.toolbar.logo=None

plot1.grid

#build select tool
pc = Select(value='Central', options=names, title='Select Territory')
#create update function
def source_update(attribute, old, new):
	n = str(pc.value)
	ds = data_slope[data_slope['Territory'] == n]
	dc = data_change[data_change['Territory'] == n]
	source_s.data = ColumnDataSource.from_df(ds)
	source_c.data = ColumnDataSource(dc).from_df(dc)
#set on change condition
pc.on_change("value",source_update)

#create save button using custom js script
button = Button(label="Save Territory", button_type="success")
button.callback = CustomJS(args=dict(source=source_c),
                           code=open(join(dirname(__file__), "download_one.js")).read())

widget1 = widgetbox(pc,width=500)
widget2 = widgetbox(button,width=500)
layout1=layout([widget1],[plot,plot1],[widget2])
curdoc().add_root(layout1)
curdoc().title = "Territory"
