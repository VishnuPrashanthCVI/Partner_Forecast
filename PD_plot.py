import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ggplot

plt.style.use('ggplot')

def pca_plot():
	#read in data
	cevr = pd.read_csv('cum_pca_exp_var.csv')
	cevr = list(cevr.iloc[:,0])
	#make data into lists
	evr = pd.read_csv('pca_exp_var.csv')
	evr = list(evr.iloc[:,0])
	x = range(1,11)

	#create figure

	fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(5,3))
	plt.rc('axes', labelsize = 8, titlesize = 8)
	plt.rc('legend', fontsize = 8)
	#plt.rc('suptitle', fontsize = 10)
	#fig.suptitle('Principal Components Analysis')

	#plot first subplot cumulative evr
	ax1.plot(x,cevr,'k',label = 'Cum Exp Var')
	ax1.legend(loc='best')
	ax1.set_title('Cumulative Explained Variance')
	ax1.set_xlabel('Number of Components')
	ax1.set_ylabel('Cum Exp Var')
	ax1.grid(True)

	#plot second subplot with explained variance by component
	ax2.plot(x,evr,'grey',label = 'Component Exp Var')
	ax2.legend(loc='best')
	ax2.set_title('Explained Variance')
	ax2.set_xlabel('Number of Components')
	ax2.set_ylabel('Exp Var by Component')
	ax2.grid(True)

	plt.tight_layout()
	plt.savefig('pca_explained_variance.png', dpi = 200)
	plt.close()

def primary_accuracy_plot():
	accuracy = pd.read_csv('primary_binary_accuracy.csv')
	acc_epochs = list(accuracy.epoch)
	accuracy = list(accuracy.acc)
	test = pd.read_csv('primary_test_scores.csv')
	test_epochs = list(test.Epochs)
	score = list(test['Test Accuracy'])

	#define figure
	fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(5,3))
	plt.rc('axes', labelsize = 6, titlesize = 8)
	plt.rc('legend', fontsize = 6)
	#plt.rc('suptitle', fontsize = 10)
	#fig.suptitle('Primary Sample Accuracy')

	#plot first subplot cumulative evr
	ax1.plot(acc_epochs,accuracy,'k',label = 'Training Accuracy', lw = .25)
	ax1.legend(loc='best')
	ax1.set_title('Training Accuracy By Epoch')
	ax1.set_ylim(bottom = 0.55,top = 1.05)
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Accuracy')
	#ax1.set_xlim((.4,1.1))
	ax1.grid(True)

	#plot second subplot with explained variance by component
	ax2.plot(test_epochs,score,'grey',label = 'Test Accuracy', lw = 1)
	ax2.legend(loc='best')
	ax2.set_title('Test Set Accuracy By Epoch')
	ax2.set_xlabel('Epochs')
	ax2.set_ylabel('Accuracy')
	ax2.set_ylim(bottom = 0.55,top = 1.05)
	ax2.grid(True)
	fig.suptitle('Primary Model Accuracy')
	plt.tight_layout(pad = 2)
	plt.savefig('primary_accuracy.png', dpi=200)
	plt.close()

def primary_precision_plot():
	perf_data = pd.read_csv('primary_performance_scores.csv')
	labels = list(perf_data.Diagnosis)
	precision = list(perf_data.Precision)
	recall = list(perf_data.Recall)
	fscore = list(perf_data.F_Score)

	fig, ax = plt.subplots(figsize=(4,2.5))
	plt.rc('axes', labelsize = 6, titlesize = 8)
	plt.rc('legend', fontsize = 6)
	width = .1
	xlocs = np.arange(len(labels))
	ax.bar(xlocs-width,precision,width,color='k',label = 'Precision')
	ax.bar(xlocs,recall,width,color='grey',label = 'Recall')
	ax.bar(xlocs+width,fscore,width,color='lightgrey',label = 'F Score')

	ax.set_xticks(ticks=range(len(labels)))
	ax.set_xticklabels(labels)
	ax.yaxis.grid(True)
	ax.legend(loc='best')
	ax.set_ylabel('Score')
	ax.set_ylim(.55,1.05)
	ax.set_xlabel('Diagnosis')
	fig.suptitle('Primary Model Scores')
	fig.tight_layout(pad=2)
	fig.savefig('primary_performance_scores.png', dpi=200)
	plt.close()

def pca_precision_plot():
	perf_data = pd.read_csv('pca_performance_scores.csv')
	labels = list(perf_data.Diagnosis)
	precision = list(perf_data.Precision)
	recall = list(perf_data.Recall)
	fscore = list(perf_data.F_Score)

	fig, ax = plt.subplots(figsize=(4,2.5))
	plt.rc('axes', labelsize = 6, titlesize = 8)
	plt.rc('legend', fontsize = 6)
	width = .1
	xlocs = np.arange(len(labels))
	ax.bar(xlocs-width,precision,width,color='k',label = 'Precision')
	ax.bar(xlocs,recall,width,color='grey',label = 'Recall')
	ax.bar(xlocs+width,fscore,width,color='lightgrey',label = 'F Score')

	ax.set_xticks(ticks=range(len(labels)))
	ax.set_xticklabels(labels)
	ax.yaxis.grid(True)
	ax.legend(loc='best')
	ax.set_ylabel('Score')
	ax.set_ylim(.45,1.05)
	fig.suptitle('Top Twenty Precision & Recall')
	fig.tight_layout(pad=2)
	fig.savefig('twenty_performance_scores.png', dpi=200)
	plt.close()

def compare_probabilities_plot():
	ppred = np.loadtxt('ppred.csv',delimiter = ',')
	pred = np.loadtxt('pred.csv', delimiter =',')
	ppred.reshape(-1,1)
	pred.reshape(-1,1)
	ppred0=ppred[ppred<.5]
	ppred1=ppred[ppred>=.5]
	pred0=pred[pred<.5]
	pred1=pred[pred>=.5]
	mean0 = [pred0.mean(),ppred0.mean()]
	mean1 = [(pred1.mean()/100),(ppred1.mean()/100)]
	std0 = [pred0.std(),ppred0.std()]
	std1 = [pred1.std(),ppred1.std()]
	labels = ['Ten PCA Components','Twenty Top Features']
	fig, ax = plt.subplots(figsize=(4,2.5))
	plt.rc('axes', labelsize = 6, titlesize = 8)
	plt.rc('legend', fontsize = 6)
	width = .1
	xlocs = np.arange(len(labels))

	ax.bar(xlocs-2*width,mean0,width,color='k',label = 'Mean No PD')
	ax.bar(xlocs-width,mean1,width,color='darkgrey',label = 'Mean PD /100')
	ax.bar(xlocs,std0,width,color='grey',label = 'Std Dev N0 PD')
	ax.bar(xlocs+width,std1,width,color='lightgrey',label = 'Std Dev PD')

	ax.set_xticks(ticks=range(len(labels)))
	ax.set_xticklabels(labels)
	ax.yaxis.grid(True)
	ax.legend(loc='best')
	ax.set_ylabel('Metric Score')
	ax.set_ylim(0,.015)
	fig.suptitle('Sample Probabilities')
	fig.tight_layout(pad=2)
	fig.savefig('probabilities.png', dpi=200)
	plt.close()

def multiple_samples_plot():
	samples = pd.read_csv('performance_over_resamples.csv')
	samples_PD = samples[samples.Diagnosis == 'PD']
	samples_noPD = samples[samples.Diagnosis == 'No_PD']

	pmean_PD = samples_PD.Precision.mean()
	pmean_noPD = samples_noPD.Precision.mean()
	pmean = [pmean_PD, pmean_noPD]
	rmean_PD = samples_PD.Recall.mean()
	rmean_noPD = samples_noPD.Recall.mean()
	rmean = [rmean_PD,rmean_noPD]

	pstd_PD = samples_PD.Precision.std()
	pstd_noPD = samples_noPD.Precision.std()
	pstd = [pstd_PD, pstd_noPD]
	rstd_PD = samples_PD.Recall.std()
	rstd_noPD = samples_noPD.Recall.std()
	rstd = [rstd_PD,rstd_noPD]

	fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(4,2.5))
	plt.rc('axes', labelsize = 10, titlesize = 10)
	plt.rc('legend', fontsize = 6)
	labels = ['PD', 'No PD']
	width = .1
	xlocs = np.arange(len(labels))

	ax1.bar(xlocs,pmean,width,color='k',label = 'Precision Mean')
	ax1.bar(xlocs+width,rmean,width,color='darkgrey',label = 'Recall Mean')
	ax1.set_xticks(ticks=range(len(labels)))
	ax1.set_xticklabels(labels)
	ax1.grid(True)
	ax1.legend(loc='best')
	ax1.set_title('Mean')
	ax1.set_ylim(bottom = 0.55,top = 1.05)
	ax1.set_xlabel('Diagnosis')
	ax1.set_ylabel('Mean')

	ax2.bar(xlocs,pstd,width,color='k',label = 'Precision STD')
	ax2.bar(xlocs+width,rstd,width,color='darkgrey',label = 'Recall STD')
	ax2.set_xticks(ticks=range(len(labels)))
	ax2.set_xticklabels(labels)
	ax2.grid(True)
	ax2.legend(loc='best')
	ax2.set_title('Std Dev')
	ax2.set_ylim(bottom = 0.0,top = .15)
	ax2.set_xlabel('Diagnosis')
	ax2.set_ylabel('STD')

	fig.suptitle('Multiple Samples Precision & Recall')
	fig.tight_layout(pad=2)
	fig.savefig('multiple_samples.png',dpi = 200)
	plt.close()

def lift_plot():
	lift_scores = pd.read_csv('lift_scores.csv')
	perf_data = pd.read_csv('primary_performance_scores.csv')
	samples = pd.read_csv('performance_over_resamples.csv')
	samples_PD = samples[samples.Diagnosis == 'PD']
	samples_noPD = samples[samples.Diagnosis == 'No_PD']
	pmean_PD = samples_PD.Precision.mean()
	pmean_noPD = samples_noPD.Precision.mean()
	prec_lift = []
	prec_samp = []
	prec_prim = []
	prec_lift.append(lift_scores.iloc[1,1])
	prec_lift.append(lift_scores.iloc[0,1])
	prec_samp.append(pmean_PD)
	prec_samp.append(pmean_noPD)
	prec_prim.append(perf_data.iloc[1,1])
	prec_prim.append(perf_data.iloc[0,1])
	labels = ['PD','NoPD']
	fig, ax = plt.subplots(figsize=(4,2.5))
	plt.rc('axes', labelsize = 10, titlesize = 10)
	plt.rc('legend', fontsize = 6)
	width = .1
	xlocs = np.arange(len(labels))
	ax.bar(xlocs-width,prec_lift,width,color='k',label = 'Absolute Precision')
	ax.bar(xlocs,prec_samp,width,color='grey',label = 'Samples Precision')
	ax.bar(xlocs+width,prec_prim,width,color='lightgrey',label = 'Primary Precision')

	ax.set_xticks(ticks=range(len(labels)))
	ax.set_xticklabels(labels)
	ax.set_xlabel('Diagnosis')
	ax.yaxis.grid(True)
	ax.legend(loc='best')
	ax.set_ylabel('Score')
	ax.set_ylim(.40,1.05)
	fig.suptitle('Lift In Precision')
	fig.tight_layout(pad=2)
	fig.savefig('lift_precision.png', dpi = 200)
	plt.close()

	rmean_PD = samples_PD.Recall.mean()
	rmean_noPD = samples_noPD.Recall.mean()
	rec_lift = []
	rec_samp = []
	rec_prim = []
	rec_lift.append(lift_scores.iloc[1,2])
	rec_lift.append(lift_scores.iloc[0,2])
	rec_samp.append(rmean_PD)
	rec_samp.append(rmean_noPD)
	rec_prim.append(perf_data.iloc[1,2])
	rec_prim.append(perf_data.iloc[0,2])
	labels = ['PD','NoPD']
	fig, ax = plt.subplots(figsize=(4,2.5))
	plt.rc('axes', labelsize = 10, titlesize = 10)
	plt.rc('legend', fontsize = 6)
	width = .1
	xlocs = np.arange(len(labels))
	ax.bar(xlocs-width,rec_lift,width,color='k',label = 'Absolute Recall')
	ax.bar(xlocs,rec_samp,width,color='grey',label = 'Samples Recall')
	ax.bar(xlocs+width,rec_prim,width,color='lightgrey',label = 'Primary Recall')

	ax.set_xticks(ticks=range(len(labels)))
	ax.set_xticklabels(labels)
	ax.set_xlabel('Diagnosis')
	ax.yaxis.grid(True)
	ax.legend(loc='best')
	ax.set_ylabel('Score')
	ax.set_ylim(.40,1.05)
	fig.suptitle('Lift In Recall')
	fig.tight_layout(pad=2)
	fig.savefig('lift_recall.png', dpi = 200)
	plt.close()





if __name__ == '__main__':
	primary_accuracy_plot()
	pca_plot()
	primary_precision_plot()
	pca_precision_plot()
	compare_probabilities_plot()
	multiple_samples_plot()
	lift_plot()
