import pandas as pd
import numpy as np
import pickle as pkl
from os.path import dirname, join
from bokeh.layouts import layout,column,row,widgetbox
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource,CustomJS,Range1d,PanTool,HoverTool,BoxZoomTool,ResetTool
from bokeh.models.widgets import Select,DataTable,TableColumn,Slider,Button
from bokeh.plotting import figure



with open('Time_Series_Predict.pkl', 'rb') as f:
	data = pkl.load(f)
#create data set for slope by new id
df = data[data.Date == '2014-11-30'].reset_index(drop=True)
df = df.sort_values(by=['Slope'],ascending=True)
#drops=[]
#drops = list(df.columns[1:7])
#drops.append(df.columns[-1])
#df = df.drop(drops,axis=1)
df['IDX'] = range(1,df.shape[0]+1)
#set source original and source to be moved with slider
source_original = ColumnDataSource(df)
source = ColumnDataSource(df)
#basic figure
plot = figure()
plot.line(x='IDX',y='Slope',source=source)
plot.plot_width=1200
plot.plot_height=650
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

#plot.x_range=Range1d(start=source.data['IDX'].min(),end = #source.data['IDX'].max())
#plot.y_range=Range1d(start= #source.data['Slope'].min(),end=source.data['Slope'].max())

plot.tools=[PanTool(),ResetTool()]
hover=HoverTool(tooltips=[('ID','@ID'),('Class','@Class'),('Territory','@Territory')])
plot.add_tools(hover)
plot.toolbar_location = 'above'
plot.toolbar.logo=None

plot.grid
plot.min_border_right = 80


def set_range(attrname,old,new):
	source.data={key:[value for i, value in enumerate(source_original.data[key]) if source_original.data['IDX'][i]<=sliderhigh.value and source_original.data['IDX'][i]>=sliderlow.value] for key in source_original.data}


#create loweer limit slider
sliderlow=Slider(start=min(source_original.data['IDX']),end=max(source_original.data['IDX']),value=1,step=10,title="Lower Partner: ")
sliderlow.on_change("value",set_range)

#create upper limit slider
sliderhigh=Slider(start=min(source_original.data['IDX'])-1,end=max(source_original.data['IDX']),value=500,step=10,title="Upper Partner: ")
sliderhigh.on_change("value",set_range)


#create save button using custom js script
button = Button(label="Save Growth Data", button_type="success")
button.callback = CustomJS(args=dict(source=source),
                           code=open(join(dirname(__file__), "download_three.js")).read())

#save revised source data ID's
#widgetbox1=widgetbox([sliderhigh,sliderlow],sizing_mode=scale_both)
#widgetbox2=WidgetBox([button],responsive=True)
widget1=widgetbox(sliderlow,sliderhigh, width = 500)
widget2=widgetbox(button, width = 500)
layout1=layout([widget1],[plot],[widget2])
curdoc().add_root(layout1)
curdoc().title = "Select Growth"
