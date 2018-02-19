import os as os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
#from fbprophet import Prophet
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import math as ma
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization,LSTM
from keras.optimizers import sgd,rmsprop,adam
from keras.initializers import RandomNormal, RandomUniform
from keras.losses import mean_squared_error, mean_absolute_error
from keras.callbacks import CSVLogger


seed = 73
np.random.seed(73)

class ModelDataLSTM():
	def __init__(self):
		pass

	def lstm_model(self, X,b=1,n=4,o=1):
		features = X.shape[2]
		model = Sequential()
		model.add(LSTM(n, batch_input_shape=(b, 1, features), stateful=True, return_sequences=True))
		model.add(LSTM(n, batch_input_shape=(b, 1, features), stateful=True, return_sequences=True))
		model.add(LSTM(n, batch_input_shape=(b, 1, features), stateful=True))
		model.add(Dense(o, activation = 'tanh'))
		model.compile(loss='mean_squared_error', optimizer='adam')
		return model

	def lstm_fit(self,model,X,Xtest,y,b=1,e=50):
		model.fit(X, y, epochs=e, batch_size=b, verbose=True, shuffle=False)
		Xpred = model.predict(X, batch_size=b)
		Xtestpred = model.predict(Xtest, batch_size=b)
		model.reset_states()
		return Xpred, Xtestpred
