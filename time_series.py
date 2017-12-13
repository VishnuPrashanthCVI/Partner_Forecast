
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle as pkl
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error as mse

seed = 73
class TimeSeries:
	def __init__(self, data):
		self.data=data
	#realighn columns and drop ID
	def time_data(self,id):
		df = self.data[self.data['ID'] == id].drop(['ID'], axis = 1, inplace=False)
		df = df.ix[:,['Date','Revenue']]
		df.columns=['ds','y']
		return df
	#fit prophet class and add 6 months
	def fit_time(self,df):
		df['y'] = np.log(df['y'])
		m = Prophet(weekly_seasonality=False, yearly_seasonality=True, changepoint_prior_scale = .05)
		m.fit(df)
		period=m.make_future_dataframe(periods=6, freq='m')
		dp=m.predict(period)
		return dp
	#recover original data and add 6 months
	def add_id_exp(self,dt):
		Mon_ID = []
		for i in range(1,301):
			Mon_ID.extend([i]*54)
		dts = pd.DataFrame()
		dts['ID'] = Mon_ID
		dts['Date'] = dt.ds
		dts['Revenue']=np.exp(dt.yhat)
		dts['Low_Revenue'] = np.exp(dt.yhat_lower)
		dts['High_Revenue'] = np.exp(dt.yhat_upper)
		return dts

	def time_error(self,dts, data):
		dte = dts[dts.Date<'2014-12-31']
		orig_mean = data.Revenue.mean()
		pred_mean = dte.Revenue.mean()
		error = mse(data['Revenue'],dte['Revenue'])
		values = [orig_mean, pred_mean, error]
		keys =  ['orig_mean', 'pred_mean', 'root_mean_error']
		time_error = dict(zip(keys,values))
		return time_error
