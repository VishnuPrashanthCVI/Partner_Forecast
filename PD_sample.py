import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, rmsprop
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping,CSVLogger
from keras import regularizers
from keras import metrics
from keras.models import load_model
from sklearn import decomposition
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_recall_fscore_support
import random as ra
from sklearn.utils import resample

#convert labels to binary and upsample
def make_binary_upsample(df,dx,dc,seed):
	df.loc[df.Dx ==  dx, 'Dx'] = 0
	for x in  dc:
		x = str(x)
		df.loc[df.Dx == x,'Dx'] = 1
	dfNP = df.loc[df.Dx == 0, :]
	dfP = df.loc[df.Dx == 1, :]
	diff1 = int(len(dfP) - len(dfNP))
	diff2 = int(len(dfNP))
	n = int(min(diff1,diff2))
	#dfN = resample(dfNP, n_samples = n, random_state =  seed, replace = False)
	dfR = resample(dfNP, n_samples = n, random_state =  seed, replace = True)
	dfb = pd.concat([dfR,df],axis=0,ignore_index=True).reset_index(drop = True)
	return dfb

#one hot encode target vector
def encode_categorical(dtrain,dtest):
	dtrain = np_utils.to_categorical(dtrain, num_classes = 2)
	dtest = np_utils.to_categorical(dtest, num_classes = 2)
	return dtrain, dtest

#compute pca component reduction
def pca_meg(dfb, seed, n=10):
	X = pd.DataFrame()
	p = []
	r = []
	evr = []
	evrc = []
	X = dfb.drop(['Dx'], axis=1).values
	X = StandardScaler().fit_transform(X)
	pca = decomposition.PCA(n_components=n, random_state = seed)
	pca.fit(X)
	C = pca.fit_transform(X)
	dim = C.shape[1]
	d = dfb.Dx.values
	d = d.reshape(-1,1)
	return C, d, dim

def split_sample(C, d, seed, s):
	Ctrain, Ctest, dtrain, dtest = train_test_split(C, d, test_size = s, random_state = seed, stratify=d)
	return Ctrain, Ctest, dtrain, dtest

#create binary keras model
def bin_model(dim, d):
	model = Sequential()
	model.add(Dense(len(d), activation='relu', input_dim=dim, kernel_initializer='normal', bias_initializer='zeros'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(200, activation='relu', kernel_initializer='normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(200, activation='relu', kernel_initializer='normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.6))
	model.add(Dense(200, activation='relu', kernel_initializer='normal'))
	model.add(BatchNormalization())
	model.add(Dropout(0.6))
	model.add(Dense(2, activation='softmax'))
	#early_stopping_monitor = EarlyStopping(patience = 3)
	adam = keras.optimizers.Adam(lr=.0001, decay = 0.01)
	#sgd = keras.optimizers.SGD()
	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	return model

#primary model for tuning hyperparameters and checking accuracy
def fit_primary_model(dim,Ctrain,Ctest,dtrain,dtest,n=500):
	modelb = bin_model(dim, dtrain)
	n_batch = Ctrain.shape[0]
	modelb.fit(Ctrain, dtrain, epochs = n, verbose = False, batch_size = n_batch, shuffle = False)
	score = modelb.evaluate(Ctest, dtest, batch_size=n_batch)
	pred = modelb.predict(Ctest, batch_size = n_batch)
	del modelb
	return pred,score

def compute_scores(y_true, y_pred):
	y_true = np.argmax(y_true,axis=1)
	y_pred = np.argmax(y_pred,axis=1)
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	diagnosis = pd.DataFrame()
	diagnosis.loc[0,'True_No_PD'] = tn
	diagnosis.loc[0,'False_No_PD'] = fn
	diagnosis.loc[0,'False_PD'] = fp
	diagnosis.loc[0,'True_PD'] = tp
	precision, recall, f_score, population = precision_recall_fscore_support(y_true, y_pred)
	performance_scores = pd.DataFrame()
	condition = ['No_PD','PD']
	performance_scores['Diagnosis'] = condition
	performance_scores['Precision'] = precision
	performance_scores['Recall'] = recall
	performance_scores['F_Score'] = f_score
	return performance_scores,diagnosis

if __name__ == '__main__':

	#read data file an set constants
	df = pd.read_csv('Phenotype_ML_All.csv')
	dx = 'C'
	dc = ['PD ','PSP','ET ']
	performance = pd.DataFrame()
	diagnosis = pd.DataFrame()
	s = .2
	for i in range(70,80):
		seed = i
		np.random.seed(seed)
		dfb = make_binary_upsample(df,dx,dc,seed)
		C, d, dim = pca_meg(dfb, seed, n=10)
		Ctrain, Ctest, dtrain, dtest = split_sample(C, d, seed, s)
		#fit model for this seed
		dtrain, dtest = encode_categorical(dtrain, dtest)
		pred,score = fit_primary_model(dim,Ctrain,Ctest,dtrain,dtest)
		#compute performance scores for this seed
		perf,diag = compute_scores(dtest,pred)
		perf['Sample'] = [i,i]
		diag.loc[0,'Sample'] = i
		performance = pd.concat([performance,perf],axis=0,ignore_index=True)
		diagnosis = pd.concat([diagnosis,diag],axis=0,ignore_index=True)
	performance.reset_index(drop=True)
	diagnosis.reset_index(drop=True)
	performance.to_csv('performance_over_resamples.csv')
	diagnosis.to_csv('diagnosis_over_resamples.csv')
