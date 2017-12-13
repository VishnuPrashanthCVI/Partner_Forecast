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
	with open('partner_data.pkl', 'rb') as f:
		data = pkl.load(f)
		return data

def dates(startm, startq, end):
	strt_qdt = datetime.strptime(startq, '%Y,%m')
	strt_mdt = datetime.strptime(startm, '%Y,%m')
	end_dt = datetime.strptime(end, '%Y,%m')
	dq = [dt for dt in rrule(MONTHLY, interval=3, dtstart=strt_qdt, until=end_dt)]
	dm = [dt for dt in rrule(MONTHLY, interval=1, dtstart=strt_mdt, until=end_dt)]
	return dq, dm

def qtr_rev(data):
	dr = pd.DataFrame()
#set revenue class by certification level
	clist =['Registered', 'Premier', 'Silver', 'Gold', 'Platinum']
	multiplier = []
	for cl in data.Class:
		idx = clist.index(cl)
		multiplier.append(((idx)*.2)+1)
	dr['Class'] = multiplier
#set revenue multiplier by territory
	multiplier = []
	tlist=['C', 'NE', 'SE', 'SW', 'W', 'NW']
	for t in data.Territory:
		idx = tlist.index(t)
		multiplier.append(((idx+.5)*.1)+1)
	dr['Territory'] = multiplier
#set seasonal variation by quarter with random variation
	multiplier = []
	qlist = [1,2,3,4]
	qwt = [.90,1.10,.90,1.20]
	qr =[.005,.015,.02,-.005,-.015,-.02]
	for qtr in data.Qtr:
		idx=qlist.index(qtr)
		s = qwt[idx]+ra.choice(qr)
		multiplier.append(s)
	dr['Season'] = multiplier
	#set mean for revenue forecast
	dr['Mean'] = data.iloc[:,9:].mean(axis=1)
	dr['Min'] = np.random.randint(3)+np.random.randn()
	dr['QtrRev']=dr.Season*dr.Mean*dr.Min
	return dr

#def mon_dates(md):
#	ID = []; Yr = []; Mon = []
#	yr = range(1,5)
#	mon = range(1,13)
#	for i in range(1,301):
#		for y in yr:
#	dm = pd.DataFrame(np.array([ID,Yr,Mon]).T, columns = ['ID','Yr','Mon'])
#	dm = dm.sort_values(by=['ID', 'Yr', 'Mon'], axis = 0).reset_index(drop=True)
#	dm['Date'] = md
#	return dm


if __name__== '__main__':
	#load data, scale, and create satisfaction scores with revenue
	data = load_data()
	#add quarterly revenue
	data['Rev'] = qtr_rev(data).QtrRev
	#create quarterly and monthly date series
	startm = '2011,1'
	startq = '2011,3'
	end = '2014,12'
	daq, dam = dates(startm,startq,end)
	#create vector of 300 quarterly and monthly dates
	Qtr = []
	Mon = []
	Mon_ID = []
	for i in range(1,301):
		Mon_ID.extend([i]*48)
		Qtr.extend(daq)
		Mon.extend(dam)
	#monthly time series data grouped by ID, Yr, Qtr, Mon
	data_by_ID = data.sort_values(by=['ID','Yr','Qtr'], axis=0)
	Partner_Qtr_Rev = pd.DataFrame()
	Partner_Qtr_Rev['ID'] = data_by_ID.ID
	Partner_Qtr_Rev['Revenue'] = data_by_ID.Rev
	Partner_Qtr_Rev['Date'] = Qtr
	#create monthly revenue data
	third = [.4,.5,.55]
	second = [.25,.28,.30]
	noise = [.001,.0015,.0020,-.001,-.0015,-.002]
	Rev = []
	for r in data_by_ID.Rev:
		x = (ra.choice(third)+ra.choice(noise))*r
		y = (ra.choice(second)+ra.choice(noise))*r
		z = r - x - y
		Rev.append(z)
		Rev.append(y)
		Rev.append(x)
	Partner_Mon_Rev = pd.DataFrame()
	Partner_Mon_Rev['ID'] = Mon_ID
	Partner_Mon_Rev['Revenue'] = Rev
	Partner_Mon_Rev['Date'] = Mon


	#filestr = 'Partner_Data_Revenue.pkl'
	#with open(filestr, 'wb') as f:
	#	pkl.dump(Partner_Qtr_Rev, f, -1)

	filestr = 'Partner_Monthly_Revenue.pkl'
	with open(filestr, 'wb') as f:
		pkl.dump(Partner_Mon_Rev, f, -1)

	filestr = 'Partner_Quarterly_Revenue.pkl'
	with open(filestr, 'wb') as f:
		pkl.dump(Partner_Qtr_Rev, f, -1)
