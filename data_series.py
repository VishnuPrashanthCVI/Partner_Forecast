import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.parser import parse
from dateutil.rrule import rrule, MONTHLY
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
import math as ma
import random as ra

def load_data():
	f = open('partner_data.pkl', 'rb')
	partner_data = pkl.load(f)
	f.close()
	return partner_data

def dates(startm, startq, end):
	strt_qdt = datetime.strptime(startq, '%Y,%m')
	strt_mdt = datetime.strptime(startm, '%Y,%m')
	end_dt = datetime.strptime(end, '%Y,%m')
	dq = [dt for dt in rrule(MONTHLY, interval=3, dtstart=strt_qdt, until=end_dt)]
	dm = [dt for dt in rrule(MONTHLY, interval=1, dtstart=strt_mdt, until=end_dt)]
	return dq, dm

def scale_resp(data):
	scaler = MinMaxScaler(feature_range = (1,5),copy=False)
	data.iloc[:,9:] = scaler.fit_transform(data.iloc[:,9:])
	return data

def qtr_rev(data):
	dr = pd.DataFrame()
#set revenue class by certification level
	clist =['Registered', 'Premier', 'Silver', 'Gold', 'Platinum']
	rank = []
	for cert in data.Cert:
		idx = clist.index(cert)
		rank.append(((idx)*.4)+1)
	dr['Cert'] = rank
#set revenue multiplier by territory
	rank = []
	tlist=['C', 'SE', 'SW', 'NW', 'NE']
	for t in data.Territory:
		idx = tlist.index(t)
		rank.append(((idx+.5)*.1)+1)
	dr['Terr'] = rank
#set seasonal variation by quarter with random variation
	rank = []
	qlist = [1,2,3,4]
	qwt = [.90,1.0,.80,1.20]
	qr =[.005,.015,.02,.025,-.005,-.015,-.02,-.025]
	for qtr in data.Qtr:
		idx=qlist.index(qtr)
		s = qwt[idx]+ra.choice(qr)
		rank.append(s)
	dr['Season'] = rank
#add random noise pattern
	noise = [1.01, .99, 1.015, .985, 1.02, .98, 1.025, .975, .97, 1.03]
	N = []
	for i in range(data.shape[0]):
		n = ra.choice(noise)
		N.append(n)
	dr['Noise'] = N
	#set mean for revenue forecast
	dr['Mean'] = data.iloc[:,10:].mean(axis=1)
	dr['Min'] = 2.0
	dr['QtrRev']=dr.Cert*dr.Terr*dr.Season*dr.Noise*dr.Mean*dr.Min
	return dr

def mon_dates(md):
	ID = []; Yr = []; Mon = []
	yr = [1,2,3]
	mon = [1,2,3,4,5,6,7,8,9,10,11,12]
	for i in range(1,301):
		for y in yr:
			for m in mon:
				ID.append(i)
				Yr.append(y)
				Mon.append(m)
	dm = pd.DataFrame(np.array([ID,Yr,Mon]).T, columns = ['ID','Yr','Mon'])
	dm = dm.sort_values(by=['ID', 'Yr', 'Mon'], axis = 0).reset_index(drop=True)
	dm['Date'] = md
	return dm


if __name__== '__main__':
	#load data, scale, and create satisfaction scores with revenue
	data = load_data()
	ds = scale_resp(data)
	dr = qtr_rev(ds)
	ds['Rev'] = dr.QtrRev
	#create quarterly and monthly date series
	startm = '2011,1'
	startq = '2011,3'
	end = '2013,12'
	daq, dam = dates(startm,startq,end)
	Q = []; M = []
	for i in range(300):
		Q = Q + daq
		M = M + dam
	#quarterly time series data grouped by Id, Yr, Qtr with Q Rev
	cols = ds.columns[3:-1]
	dt = pd.DataFrame(ds.drop(cols, inplace = False, axis = 1))
	dt = dt.sort_values(by=['ID','Yr','Qtr'], axis = 0).reset_index(drop=True)
	dt['Date'] = Q
	#monthly time series data grouped by ID, Yr, Qtr, Mon
	dm = mon_dates(M)
	third = [.5,.55,.6]
	second = [.25,.30,.35]
	noise = [.01,.015,.020,-.01,-.015,-.2]
	Rev = []
	for q in dt.Rev:
		x = (ra.choice(third)+ra.choice(noise))*q
		y = (ra.choice(second)+ra.choice(noise))*q
		z = q - x - y
		Rev.append(z)
		Rev.append(y)
		Rev.append(x)
	dm['Rev'] = Rev

	filestr = 'Rank_Data_Scaled.pkl'
	f = open(filestr, 'wb')
	pkl.dump(ds, f, -1)
	f.close()

	filestr = 'Qtr_Rev_Data.pkl'
	f = open(filestr, 'wb')
	pkl.dump(dt, f, -1)
	f.close()

	filestr = 'Mon_Rev_Data.pkl'
	f = open(filestr, 'wb')
	pkl.dump(dm, f, -1)
	f.close()
