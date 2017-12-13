import os as os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import math as ma
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import adam,sgd


model = Sequential()
model.add(LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2])))
#model.add(Dropout(.5))
#model.add(LSTM(50,return_sequences=True,stateful=True))
#model.add(Dropout(.5))
#model.add(LSTM(50,return_sequences=True,stateful=True))
#model.add(Dropout(.5))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs = 50, batch_size = 32, shuffle = False)
