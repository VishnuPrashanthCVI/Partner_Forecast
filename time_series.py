import numpy as np
from datetime import datetime, timedelta
import pickle as pkl
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error as mse
import pandas as pd


seed = 73
np.random.seed(seed)

class TimeSeries:
	def __init__(self,data):
		self.data = data
	#realighn columns and drop ID
	def time_data(self, id):
		df = self.data[self.data['ID'] == id]
		df = df.loc[df.Date<'2014-10-01',:]
		df = df.loc[:,['Date','Revenue']]
		df.columns=['ds','y']
		return df
	#fit prophet class and add 6 months
	def fit_time(self,df):
		df['y'] = np.log(df['y'])
		m = Prophet(weekly_seasonality=False, yearly_seasonality=True, changepoint_prior_scale = .05)
		m.fit(df)
		period=m.make_future_dataframe(periods=3, freq='m')
		dp=m.predict(period)
		return dp

	def time_error(self,dt):
		orig_mean = self.data.Revenue.mean()
		pred_mean = dt.Pred_Revenue.mean()
		error = mse(self.data['Revenue'],dt['Pred_Revenue'])
		values = [orig_mean, pred_mean, error]
		keys =  ['orig_mean', 'pred_mean', 'root_mean_error']
		time_error = dict(zip(keys,values))
		return time_error
