import os as os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from fbprophet import Prophet
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import math as ma
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import sgd,rmsprop,adam
from keras.initializers import RandomNormal, RandomUniform
from keras.losses import mean_squared_error, mean_absolute_error
import keras
from keras.callbacks import CSVLogger

from prep_data_NN import PrepDataNN
from multivariate_nn import MultiVariate

seed = 73
np.random.seed(73)

#fetch data
with open('Part_Qtr_Rev_Data.pkl', 'rb') as f:
	data = pkl.load(f)
#instantiate data prep class
prep = PrepDataNN(data)
#prepare data for multivariate neural network
x_train,y_train,x_pred,y_pred = prep.prep_data_NN()
#instantiate multivariate nn class
mv = MultiVariate()
model = mv.sequential_model(x_train)
yhat = mv.model_fit(x_train,y_train,x_pred,y_pred, model)
pred = pd.DataFrame()
pred['PV'] = yhat
pred['AV'] = ypred
pred.to_csv('MV_NN_Prediction.csv')
error = mse(yhat,y_pred)
