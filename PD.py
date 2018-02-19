import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, rmsprop
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import CSVLogger
from keras import regularizers
from keras import metrics
from keras.models import load_model
from sklearn import decomposition
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

#compute pca component reduction
def pca_meg(dfb, seed, n=10):
	X = pd.DataFrame()
	p = []
	r = []
	evr = []
	evrc = []
	X = dfb.drop(['Dx'], axis=1).values
	#X = StandardScaler().fit_transform(X)
	X = MinMaxScaler(feature_range = (0,1)).fit_transform(X)
	pca = decomposition.PCA(n_components=n, random_state = seed)
	pca.fit(X)
	#compute explained variance by component
	evr = pca.explained_variance_ratio_
	evr=pd.DataFrame(evr)
	evr.columns=['Var Ratio']
	evr.to_csv('pca_exp_var.csv',index=False)
	#compute cumulative explained variance
	evrc = pca.explained_variance_ratio_.cumsum()
	evrc=pd.DataFrame(evrc)
	evrc.columns=['Cum Var Ratio']
	evrc.to_csv('cum_pca_exp_var.csv',index=False)
	p = pca.components_
	locations = df.drop('Dx',axis=1).columns
	p = pd.DataFrame(p)
	p.columns = locations
	p.to_csv('pca_weights.csv',index=False)
	C = pca.fit_transform(X)
	dim = C.shape[1]
	d = dfb.Dx.values
	d = d.reshape(-1,1)
	return C, d, dim
#C is matrix of decomposed features, d the binary target vector and dim is samples dimension

#stratify and split sample for testing
def split_sample(C, d, seed, s):
	Ctrain, Ctest, dtrain, dtest = train_test_split(C, d, test_size = s, random_state = seed, stratify = d)
	return Ctrain, Ctest, dtrain, dtest

#one hot encode target vector
def encode_categorical(dtrain,dtest):
	dtrain = np_utils.to_categorical(dtrain, num_classes = 2)
	dtest = np_utils.to_categorical(dtest, num_classes = 2)
	return dtrain, dtest

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
def fit_primary_model(dim,Ctrain,Ctest,dtrain,dtest,n=500, name ='primary'):
	modelb = bin_model(dim, dtrain)
	n_batch = Ctrain.shape[0]
	epoch_log = CSVLogger(name + '_binary_accuracy.csv')
	#early_stop = EarlyStopping(monitor='loss', patience=2, verbose=0)
	callbacks = [epoch_log]
	modelb.fit(Ctrain, dtrain, epochs = n, verbose = True, batch_size = n_batch, callbacks = callbacks)
	score = modelb.evaluate(Ctest, dtest, batch_size=n_batch)
	accuracy=pd.read_csv(name + '_binary_accuracy.csv')
	pred = modelb.predict(Ctest, batch_size = n_batch)
	weights = modelb.get_weights()
	modelb.save(name + '_model.h5')
	del modelb
	return pred,accuracy,score,weights

#model for estimating test score  by epoch
def fit_score_model(dim,Ctrain,Ctest,dtrain,dtest,n,weights):
	models = bin_model(dim, dtrain)
	models.set_weights(weights)
	n_batch = Ctrain.shape[0]
	models.fit(Ctrain, dtrain, epochs=n, verbose = False, batch_size = n_batch, shuffle = False)
	test_score = models.evaluate(Ctest, dtest)[1]
	del models
	return test_score

def compute_primary_scores(dim,Ctrain,Ctest,dtrain,dtest,weights,n1=25,n2=525,n3=25, name = 'primary'):
	test_score = 0
	scores=[]
	#compute scores for epoch iterations
	for n in range(n1,n2,n3):
		test_score = fit_score_model(dim,Ctrain,Ctest,dtrain,dtest,n,weights)
		scores.append(test_score)
	#plot scores
	x = range(n1,n2,n3)
	#save dataframe of scores
	epoch_scores = pd.DataFrame()
	epoch_scores['Epochs'] = x
	epoch_scores['Test Accuracy'] = scores
	epoch_scores.to_csv(name  + '_test_scores.csv', index = False)
	return scores

#select most imortant features in pca analysis using coefficients weights by explained variance
def pca_importance(dfb, n=20):
	pca_wgt = pd.read_csv('pca_weights.csv')
	locations = pca_wgt.columns
	pca_evr = pd.read_csv('pca_exp_var.csv')
	coeffs = np.absolute(np.array(pca_wgt)*np.array(pca_evr))
	imp_weights = []
	for i in range(coeffs.shape[1]):
		coeff = coeffs[:,i].sum()
		imp_weights.append(coeff)
	pca_imp = pd.DataFrame()
	pca_imp['Locations'] = locations
	pca_imp['Coefficients'] = imp_weights
	pli = pca_imp.sort_values(by='Coefficients', ascending = False).head(n)
	pli.to_csv('pca_locations_importance.csv', index = False)
	cols = list(pli.Locations)
	dfp = dfb[cols]
	return dfp

def prepare_pca_model_data(dfp, d, seed):
	C = StandardScaler().fit_transform(dfp)
	dim = C.shape[1]
	s = .2
	Ptrain, Ptest, ptrain, ptest = split_sample(C, d, seed, s)
	ptrain, ptest = encode_categorical(ptrain,ptest)
	return dim, Ptrain, Ptest, ptrain, ptest

def compute_scores(y_true, y_pred, name = 'trial'):
	y_true = np.argmax(y_true,axis=1)
	y_pred = np.argmax(y_pred,axis=1)
	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
	diagnosis = pd.DataFrame()
	diagnosis.loc[0,'True_No_PD'] = tn
	diagnosis.loc[0,'False_No_PD'] = fn
	diagnosis.loc[0,'False_PD'] = fp
	diagnosis.loc[0,'True_PD'] = tp
	diagnosis.to_csv(name + '_diagnosis.csv', index = False)
	precision, recall, f_score, population = precision_recall_fscore_support(y_true, y_pred)
	performance_scores = pd.DataFrame()
	condition = ['No PD','PD']
	performance_scores['Diagnosis'] = condition
	performance_scores['Precision'] = precision
	performance_scores['Recall'] = recall
	performance_scores['F_Score'] = f_score
	performance_scores.to_csv(name + '_performance_scores.csv', index = False)
	return performance_scores,diagnosis

def compute_lift_scores(dtest):
	ltest = np.argmax(dtest, axis=1)
	pd = float(ltest.sum())
	nopd = float(len(ltest) - pd)
	denom = float(len(ltest))
	#all 0 or all No PD
	nopd_precision = nopd/denom
	nopd_recall = nopd/denom
	#all 1 or PD
	pd_precision = pd/denom
	pd_recall = pd/denom
	precision = []
	precision.append(nopd_precision)
	precision.append(pd_precision)
	recall = []
	recall.append(nopd_recall)
	recall.append(pd_recall)
	#lift_score = pd.DataFrame()
	lift_score['Score'] = ['NoPD', 'PD']
	lift_score['Precision'] = precision
	lift_score['Recall'] = recall
	lift_score.to_csv('lift_scores.csv',index = False)
	return lift_score

if __name__ == '__main__':
	#read data file an set constants
	df = pd.read_csv('Phenotype_ML_All.csv')
	seed = 73
	np.random.seed(seed)
	dx = 'C'
	dc = ['PD ','PSP','ET ']
	#prepare binary data, upsample control group, and decompose features
	#only upsample  by 50% of smallest class
	dfb = make_binary_upsample(df,dx,dc,seed)
	#compute pca compoenets and convert d to numpy vector
	C, d, dim = pca_meg(dfb, seed, n=10)
	#split sample in stratified samples
	#portion of sample used in test set is s
	s = .2
	Ctrain, Ctest, dtrain, dtest = split_sample(C, d, seed, s)
	#one hot encode for nn fit
	dtrain, dtest = encode_categorical(dtrain,dtest)
	#compute primary model and test scores for different epochs
	pred,accuracy,score,weights = fit_primary_model(dim,Ctrain,Ctest,dtrain,dtest)
	scores = compute_primary_scores(dim,Ctrain,Ctest,dtrain,dtest,weights)
	#compute most important features by pca weights
	dfp = pca_importance(dfb)
	#test pca most important features on primary model structure
	dim, Ptrain, Ptest, ptrain, ptest = prepare_pca_model_data(dfp, d, seed)
	#compute accuracy and test accuracy on pca most important features
	ppred, paccuracy, pscore, pweight = fit_primary_model(dim, Ptrain, Ptest, ptrain, ptest, n = 500, name = 'pca')
	primary_performance,primary_diagnosis = compute_scores(dtest,pred,name='primary')
	pca_performance,pca_diagnosis = compute_scores(ptest,ppred,name = 'pca')
	#save files for probability plotting
	np.savetxt('pred.csv', pred, delimiter = ',')
	np.savetxt('ppred.csv', ppred, delimiter = ',')
	#create pred vector for measuring lift
	#pred1 = np.ones(len(pred))
	#pred0 = np.zeros(len(pred))
	lift_score=pd.DataFrame()
	lift = compute_lift_scores(dtest)
