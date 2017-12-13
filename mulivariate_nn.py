from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import sgd,rmsprop
from keras.initializers import RandomNormal, RandomUniform
from keras.losses import mean_squared_error, mean_absolute_error
import keras
import os as os
import pandas as pd
import numpy as np


class  MultiVariate:
	def __init__(self):
		pass

	def model_sequential(self,x_train,x_test,y_train):
		model = Sequential()
		model.add(Dense(64, input_dim = (x_train.shape[1]), activation = 'relu', kernel_initializer = 'RandomNormal'))
		model.add(Dropout(.4))
		model.add(Dense(64, activation = 'relu'))
		model.add(Dropout(.4))
		model.add(Dense(64, activation = 'relu'))
		model.add(Dropout(.4))
		model.add(Dense(1, activation = 'relu'))
		model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
		model.fit(x_train,y_train,batch_size=128,epochs=64,validation_split=.1)
		yhat = model.predict(x_test)
		return yhat
