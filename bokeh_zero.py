import pandas as pd
import numpy as np
import pickle as pkl
from os.path import dirname, join
from bokeh.layouts import layout,column,row,widgetbox
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource,PanTool,ResetTool,CustomJS
from bokeh.models.widgets import Select,Button
from bokeh.plotting import figure

with open('Time_Series_Predict.pkl', 'rb') as f:
	data = pkl.load(f)

#create slope data set using idx
data_last = data[data.Date == '2014-11-30'].reset_index(drop=True)
data_slope = data_last.sort_values(by=['Slope'],ascending=True)
data_slope['IDX'] = range(1,data_slope.shape[0]+1)
data_change = data_last.sort_values(by=['Change'],ascending=True)
data_change['IDX'] = range(1,data_change.shape[0]+1)
#set source original for slope and change
names = ['Registered','Premier','Silver','Gold','Platinum']

ds = data_slope[data_slope.Class == names[0]]
dc = data_change[data_change.Class == names[0]]

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
plot.min_border_right = 80
plot.min_border_left = 80

plot.tools=[PanTool(),ResetTool()]
hover=HoverTool(tooltips=[('ID','@ID'),('Class','@Class'),('Territory','@Territory')])
plot.add_tools(hover)
plot.toolbar_location = 'above'
plot.toolbar.logo=None

#basic figure for change
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

plot1.tools=[PanTool(),ResetTool()]
hover=HoverTool(tooltips=[('ID','@ID'),('Class','@Class'),('Territory','@Territory')])
plot1.add_tools(hover)
plot1.toolbar_location = 'above'
plot1.toolbar.logo=None

plot1.border_fill_color='whitesmoke'
plot1.grid
plot1.min_border_right = 80
plot1.min_border_left = 80
#build select tool
pc = Select(value=names[0], options=names, title='Select Class')
#create update function
def source_update(attribute, old, new):
	n = str(pc.value)
	ds = data_slope[data_slope['Class'] == n]
	dc = data_change[data_change['Class'] == n]
	source_s.data = ColumnDataSource.from_df(ds)
	source_c.data = ColumnDataSource(dc).from_df(dc)
#set on change condition
pc.on_change("value",source_update)

#create save button using custom js script
button = Button(label="Save Class", button_type="success")
button.callback = CustomJS(args=dict(source=source_c),
                           code=open(join(dirname(__file__), "download_zero.js")).read())

widget1 = widgetbox(pc,width=500)
widget2 = widgetbox(button,width=500)
layout1=layout([[widget1],[plot,plot1],[widget2]])
curdoc().add_root(layout1)
curdoc().title = "Class"





'''#create columnsource for growth magnitude by IDX and Class

df = data[data.Date == '2014-11-30'].reset_index(drop=True)
df = df.sort_values(by=['Change'],ascending=True)
df['IDX'] = range(1,df.shape[0]+1)
#set source original and source to be moved with slider




#basic figure
plot1 = figure()
plot1.line(x='IDX',y='Change',line_color = 'blue', legend = 'Registered', source=source)

plot1.plot_width=1200
plot1.plot_height=650
plot1.background_fill_color="lightblue"
plot1.background_fill_alpha=0.3

plot1.title.text="Absolute Change By Class"
plot1.title.text_color="black"
plot1.title.text_font="Arial"
plot1.title.text_font_size="28px"
plot1.title.align="center"

plot1.xaxis.axis_label="Partners"
plot1.yaxis.axis_label="Absolute Change"
plot1.axis.axis_label_text_font="Arial"
plot1.axis.axis_label_text_font_size="18px"

plot1.border_fill_color='whitesmoke'

#plot1.x_range=Range1d(start=source.data['IDX'].min(),end = #source.data['IDX'].max())
#plot1.y_range=Range1d(start= #source.data['Slope'].min(),end=source.data['Slope'].max())

plot1.tools=[PanTool(),ResetTool()]
plot1.toolbar_location = 'above'
plot1.toolbar.logo=None

plot1.grid
plot1.min_border_right = 80
plot1.legend.location = 'top_left'

layout1=column([Class,plot,plot1])
curdoc().add_root(layout1)
curdoc().title = "Class"


def set_range(attrname,old,new):
	source.data={key:[value for i, value in enumerate(source_original.data[key]) if source_original.data['IDX'][i]<=sliderhigh.value and source_original.data['IDX'][i]>=sliderlow.value] for key in source_original.data}


#create loweer limit slider
sliderlow=Slider(start=min(source_original.data['IDX']),end=max(source_original.data['IDX']),value=1,step=10,title="Lower Partner: ")
sliderlow.on_change("value",set_range)

#create upper limit slider
sliderhigh=Slider(start=min(source_original.data['IDX'])-1,end=max(source_original.data['IDX']),value=500,step=10,title="Upper Partner: ")
sliderhigh.on_change("value",set_range)


##create loweer limit slider
#sliderlow=Slider(start=min(source.data['IDX']),end=max(source.data['IDX']),value#=min(source.data['IDX']),step=10,title="Lower Partner: ")
#sliderlow.on_change("value",set_range)
#
##create upper limit slider
#sliderhigh=Slider(start=min(source.data['IDX']),end=max(source.data['IDX']),valu#e=max(source.data['IDX']), step=10,title="Upper Partner: ")
#sliderhigh.on_change("value",set_range)


#create save button using custom js script
button = Button(label="Save", button_type="success")
button.callback = CustomJS(args=dict(source=source),
                           code=open(join(dirname(__file__), "download_one.js")).read())

#save revised source data ID's
#widgetbox1=widgetbox([sliderhigh,sliderlow],sizing_mode=scale_both)
#widgetbox2=WidgetBox([button],responsive=True)
#widget1=widgetbox(sliderlow,sliderhigh, width = 400)
#widget2=widgetbox(button, width = 400)'''
#layout1=layout([[plot]])
#curdoc().add_root(layout1)
#curdoc().title = "Growth Class"
