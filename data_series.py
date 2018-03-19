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
	with open('true_partner_data.pkl', 'rb') as f:
		data = pkl.load(f)
	data_desc = data.loc[:,data.columns[:11]]
	data_desc.drop(['Response_Number', 'Respondent', 'Credit_Rating'], axis =1, inplace = True)
	data_desc = data_desc.drop_duplicates(['Yr','Qtr','ID']).reset_index(drop=True)
	data_rate = data.groupby(['Yr', 'Qtr', 'ID']).mean()
	data_rate = data_rate.drop(['Response_Number'],axis=1)
	data_rate.reset_index(inplace=True)
	#merge registration data with prepared data
	data = pd.merge(data_desc, data_rate, on = ['Yr','Qtr','ID'])
	data = data.sort_values(by = ['ID','Yr','Qtr'],axis=0)
	data = data.reset_index(drop=True)
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
	dr['ID'] = data.ID
	dr['Yr'] = data.Yr
	dr['Qtr'] = data.Qtr
#set revenue class by certification level
	clist =['Registered', 'Premier', 'Silver', 'Gold', 'Platinum']
	minlist =[1,2,3,4,5,6]
	multiplier = []
	for i in range(len(data.ID)):
		if i == 0:
			idx = clist.index(data.Class[i])
			m = range(minlist[idx],minlist[idx+1])
			m = ra.choice(m)
			multiplier.append(m)

		elif (i>0) and i<len(data.ID):
			if data.Class[i-1] == data.Class[i]:
				m = multiplier[i-1]
				multiplier.append(m)
			else:
				idx = clist.index(data.Class[i])
				m = range(minlist[idx],minlist[idx+1])
				m = ra.choice(m)
				multiplier.append(m)
		else:
			m = multiplier[i-1]
			multiplier.apppend(m)

	dr['Start'] = multiplier

#set seasonal variation by quarter with random variation
	multiplier = []
	qlist = [1,2,3,4]
	qwt = [.95,1.05,.90,1.10]
	qr =[.015,.010,.005,-.005,-.010,-.015]
	for qtr in data.Qtr:
		idx=qlist.index(qtr)
		s = qwt[idx]+ra.choice(qr)
		multiplier.append(s)
	dr['Season'] = multiplier
	#set mean for revenue forecast
	dr['Mean'] = data.iloc[:,9:].mean(axis=1)
	dr['QtrRev']=dr.Mean*dr.Start*dr.Season
	return dr

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
	#create vector of quarterly and monthly dates
	Qtr = []
	Mon = []
	MID = []
	MT = []
	MC = []
	for i in range(data.ID.min(),data.ID.max()+1):
		MID.extend([i]*len(dam))
		mt = [data[data['ID'] == i].iloc[0,7] for j in range(len(dam))]
		mc = [data[data['ID'] == i].iloc[0,3] for j in range(len(dam))]
		MT.extend(mt)
		MC.extend(mc)
		Qtr.extend(daq)
		Mon.extend(dam)
	#monthly time series data grouped by ID, Yr, Qtr, Mon
	data['Date'] = Qtr
	data = data.sort_values(by=['ID','Date'], axis=0)
	data = data.reset_index(drop=True)
	#create monthly revenue data time series
	third = [.4,.43]
	second = [.3,.33]
	noise = [.001,.0015,.0020,-.001,-.0015,-.002]
	Rev = []
	for r in data.Rev:
		x = (ra.choice(third)+ra.choice(noise))*r
		y = (ra.choice(second)+ra.choice(noise))*r
		z = r - x - y
		Rev.append(z)
		Rev.append(y)
		Rev.append(x)
	Partner_Mon_Rev = pd.DataFrame()
	Partner_Mon_Rev['ID'] = MID
	Partner_Mon_Rev['Revenue'] = Rev
	Partner_Mon_Rev['Date'] = Mon
	Partner_Mon_Rev['Territory'] = MT
	Partner_Mon_Rev['Class'] = MC


	filestr = 'Partner_Monthly_Revenue.pkl'
	with open(filestr, 'wb') as f:
		pkl.dump(Partner_Mon_Rev, f, -1)

	filestr = 'Part_Qtr_Rev_Data.pkl'
	with open(filestr, 'wb') as f:
		pkl.dump(data, f, -1)
