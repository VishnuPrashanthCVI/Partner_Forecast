#form 3d matrices required by LSTM requirements of time sequence steps
x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
#y_train = y_train.reshape(y_train.shape[0],1,1)
#y_test = y_test.reshape(y_test.shape[0],1,1)
#build neuarl model in keras
'''model = Sequential()
model.add(LSTM(50, batch_input_shape=(32, x_train.shape[1], x_train.shape[2]),return_sequences=True,stateful=True))
model.add(Dropout(.5))
model.add(LSTM(50,return_sequences=True,stateful=True))
model.add(Dropout(.5))
model.add(LSTM(50,return_sequences=True,stateful=True))
model.add(Dropout(.5))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs = 50, batch_size = 32)'''
