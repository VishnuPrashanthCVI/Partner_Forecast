import os as os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer
import math as ma

from sparse_data import SparseData
from time_series import TimeSeries
from prep_data import PrepData
from multivariate_nn import MultiVariate

#load and strip pickle file
with open ('raw_partner_data.pkl','rb') as f:
	data_raw = pkl.load(f)
#call sparsedata class
fd = SparseData(data_raw)

#first step in factorizations is melting dictionaries of training and prediction data
X, y, Xpred, data_train, data_predict, data_cols, data_reg, data_resp = fd.melt_dict()
#apply matrix factorization to form predictions
y_pred, mse = fd.apply_fm(X, y, Xpred)
#merge responses with predicted responses and form filled data set dfp
df = fd.merge_data(data_predict, y_pred, data_train, data_reg)
#pickle new filled data for additional analysis
with open ('partner_data.pkl','wb') as f:
	pkl.dump(df, f,-1)
#read in monthly revenue by partner and month
with open('Partner_Monthly_Revenue.pkl', 'rb') as f:
	data = pkl.load(f)
#initiate time series class
ts = TimeSeries(data)
#use times series to predict 6 months time frame
dt = pd.DataFrame()
#forecast for each id and montly series
for id in range(1,301):
	df=ts.time_data(id)
	dp=ts.fit_time(df)
	dt = pd.concat([dt,dp],axis=0,ignore_index=True)
#result in dt is now 54 periods long and the id column is missing
#need to apply np.exp and new id columns
dts = ts.add_id_exp(dt)
#calculate mean and error into dictionary
time_error = ts.time_error(dts,data)
#sve results
filestr = 'Time_Series_Predict.pkl'
with open(filestr, 'wb') as f:
	pkl.dump(dts, f, -1)
#save time series error
filestr = 'Time_Series_Error.pkl'
with open(filestr, 'wb') as f:
	pkl.dump(time_error, f, -1)
#clear some memory and remove objects to be reused
data = None
dts = None
dt = None
df = None
dp = None
X = None
y = None
Xpred = None
data_train = None
data_predict = None
data_cols = None
data_reg = None
data_resp = None
mse = None
#load quarterly revenue as data
'''with open('Partner_Quarterly_Revenue.pkl', 'rb') as f:
	dpr = pkl.load(f)
#load partner data as df
with open('partner_data.pkl', 'rb') as f:
	dpd = pkl.load(f)
#instantiate data prep class
prep = PrepData()
#prepare data for neural network
x_train, y_train, x_test, y_test = prep.prep_data(dpr,dpd)
#instantiate multivariate nn class
mv = MultiVariate()
yhat = mv.model_sequential(x_train,x_test,y_train)'''
