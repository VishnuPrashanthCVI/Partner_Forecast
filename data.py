import pandas as pd
import numpy as np
import random as ra
from sklearn.utils import resample
import pickle as pkl


#create partner index
def part_regis(n, q, yr):
	ds = pd.DataFrame()
	ds['ID'] = range(1,n + 1)
	#create partner certification
	clist=['Platinum','Gold','Silver','Premier', 'Registered']
	Cert = []
	for i in range(n):
		cert = ra.choice(clist)
		Cert.append(cert)

	#create partner primary channel
	dlist=['VAR', 'DVAR', 'DIST']
	RTM = []
	for i in range(n):
		rtm = ra.choice(dlist)
		RTM.append(rtm)

	#create primary customer target
	pclist=['ENT', 'ENT', 'CAR', 'CAR', 'ENT', 'GOVT', 'ENT', 'EDUC']
	Cust = []
	for i in range(n):
		cust = ra.choice(pclist)
		Cust.append(cust)

	#create credit rating
	credit = range(4,11)
	Credit = []
	for i in range(n):
		cr = ra.choice(credit)
		Credit.append(cr)

	#create customer target size
	cslist=['10','100','250','1000','10000','25000']
	Cust_Size = []
	for i in range(n):
		cust = ra.choice(cslist)
		Cust_Size.append(cust)

	#create territory
	tlist=['NE', 'SE', 'SW', 'NW', 'C']
	Terr = []
	for i in range(n):
		terr = ra.choice(tlist)
		Terr.append(terr)

	#create basic data file
	ds['Yr'] = yr
	ds['Qtr'] = q
	ds['Cert'] = Cert
	ds['RTM'] = RTM
	ds['Cust'] = Cust
	ds['Cust_Size'] = Cust_Size
	ds['Credit'] = Credit
	ds['Territory'] = Terr
	ds['Yr'] = yr
	ds['Qtr'] = q
	filestr = 'o_partner_reg_data_' + str(yr) + '_' + str(q) +'.pkl'
	f = open(filestr, 'wb')
	pkl.dump(ds,f,-1)
	f.close()
	ds = pd.DataFrame()

#create company rankings of partner

def mgt_ratings(n, q, yr, rmlist):
	ds = pd.DataFrame()
	ds['ID'] = range(1,n+1)
	cols = ['Sales','Engr','Training','Support','Operations','Expert']
	for i in range(len(cols)):
		rml = []
		rml = rmlist[i]
		Resp = []
		for j in range(ds.shape[0]):
			resp = ra.choice(rml)
			Resp.append(resp)
		ds[cols[i]] = Resp
	filestr = 'o_mgt_ratings_' + str(yr) + '_' + str(q) + '.pkl'
	f = open(filestr, 'wb')
	pkl.dump(ds,f,-1)
	f.close()


def part_web_act(n, q, yr, act1, act2):
	ds = pd.DataFrame()
	ds['ID'] = range(1,n+1)
	cols = ['Bids','Quotes','Registrations','Inquiries','Sessions']
	for col in cols:
		Resp = []
		for i in range(ds.shape[0]):
			resp = ra.randint(act1,act2)
			Resp.append(resp)
		ds[col] = Resp
	filestr = 'o_web_activity_' + str(yr) + '_' + str(q) + '.pkl'
	f = open(filestr, 'wb')
	pkl.dump(ds,f,-1)
	f.close()
	ds = pd.DataFrame()


#create multiple respondents of exec, sales, operations, staff
#note resample is a numpy utility that works for pandas also
def part_satisfaction(n, r, yr, q, rlist):
	ds=pd.DataFrame()
	ds['ID'] = range(1,n+1)
	dss = resample(ds,n_samples=r,random_state=71)
	ds = pd.concat([ds,dss], axis = 0, ignore_index = True)
	slist=['Exec', 'Exec', 'Sales','Sales','Sales','Opns','Staff']
	Resp = []
	for i in range(ds.shape[0]):
		resp = ra.choice(slist)
		Resp.append(resp)
	ds['Resp'] = Resp
	ds.sort_values(by = 'ID', ascending=True, inplace=True)

#create general satisfaction questions
	questions = ['Satisfaction', 'Value_Potential', 'Expected_Relationship_Duration', 'Profit_Potential', 'Product_Quality', 'Product_Importance', 'Product_Breadth', 'Product_Competitiveness', 'Product_Training', 'Customer_Referral', 'Profit_Expectation', 'Effective_Communications', 'Price_Point', 'Margin', 'Brand_Requirement', 'Training_Effectiveness', 'Customer_Recognition', 'Problem_Solving']

	for i  in range(len(questions)):
		Resp = []
		rsl = []
		rsl = rlist[i]
		for j in range(ds.shape[0]):
			resp = ra.choice(rsl)
			Resp.append(resp)
		ds[questions[i]] = Resp
	filestr = 'o_partner_sat_data' + str(yr) + '_' + str(q) + '.pkl'
	f = open(filestr, 'wb')
	pkl.dump(ds,f,-1)
	f.close()


#create account team ratings
def part_team_sat(n, r, yr, q, rlist):
	ds=pd.DataFrame()
	ds['ID'] = range(1,n+1)
	dss = resample(ds,n_samples=r,random_state=71)
	ds = pd.concat([ds,dss], axis = 0, ignore_index = True)
	slist=['Exec', 'Exec', 'Sales','Sales','Sales','Opns','Staff']
	Resp = []
	for i in range(n+r):
		resp = ra.choice(slist)
		Resp.append(resp)
	ds['Resp'] = Resp
	ds.sort_values(by = 'ID', ascending=True, inplace=True)
	Resp = []

	questions = ['Acct_Mgr', 'Support_Mgr', 'Team_Ability', 'Team_Contribution', 'Operations_Ability', 'Acct_Team_Satisfaction', 'Team_Solutions', 'Overall_Satisfaction']
	for i  in range(len(questions)):
		Resp = []
		rsl = []
		rsl = rlist[i]
		for j in range(ds.shape[0]):
			resp = ra.choice(rsl)
			Resp.append(resp)
		ds[questions[i]] = Resp
	filestr = 'o_part_team_rating'+str(yr)+'_'+str(q)+'.pkl'
	f = open(filestr, 'wb')
	pkl.dump(ds,f,-1)
	f.close()


def part_web_sat(n, r, yr, q, rlist):
	ds=pd.DataFrame()
	ds['ID'] = range(1,n+1)
	dss = resample(ds,n_samples=r,random_state=71)
	ds = pd.concat([ds,dss], axis = 0, ignore_index = True)
	slist=['Engr', 'Sales', 'Sales','Opns','Staff']
	Resp = []
	for i in range(n+r):
		resp = ra.choice(slist)
		Resp.append(resp)
	ds['Resp'] = Resp
	ds.sort_values(by = 'ID', ascending=True, inplace=True)
	questions = ['Usefulness', 'Effectiveness', 'Response', 'Error_Free', 'Ease_of_Use', 'Pricing', 'Delivery_Response', 'Overall_Satisfaction']
	for i  in range(len(questions)):
		Resp = []
		rsl = []
		rsl = rlist[i]
		for j in range(ds.shape[0]):
			resp = ra.choice(rsl)
			Resp.append(resp)
		ds[questions[i]] = Resp
	filestr = 'o_part_web_rating'+str(yr)+'_'+str(q)+'.pkl'
	f = open(filestr, 'wb')
	pkl.dump(ds,f,-1)
	f.close()

if __name__ == '__main__':

	n = 300
	r = 100
	qtrs = [1,2,3,4]
	yrs= [1,2,3]
	rlist = []
	rmlist = []
	rl = []
	rml = []
	for yr in yrs:
		for q in qtrs:
			if yr == 1 and q == 1:
				rlist = []
				rmlist = []
				rl = []
				p = [np.nan,3,3,2,2,2,2,1,1,1]
				for i in range(20):
					for j in range(8):
						ps = ra.choice(p)
						rl.append(ps)
					rlist.append(rl)
					rl = []
				act1=10;act2=50
				pr = [4,3,3,3,3,2,2,1]
				for i in range(10):
					for j in range(8):
						rp = ra.choice(pr)
						rml.append(rp)
					rmlist.append(rml)
					rml = []
			if yr == 1 and q == 2:
				rlist = []
				rmlist = []
				p = [np.nan,4,4,3,3,2,2,2,2,1,1]
				for i in range(20):
					for j in range(8):
						ps = ra.choice(p)
						rl.append(ps)
					rlist.append(rl)
					rl = []
				act1=15;act2=60
				pr = [4,4,3,3,3,2,2,1]
				for i in range(10):
					for j in range(8):
						rp = ra.choice(pr)
						rml.append(rp)
					rmlist.append(rml)
					rml = []
			if yr == 1 and q == 3:
				rlist = []
				rmlist = []
				rl = []
				p = [np.nan,4,4,3,3,3,3,2,2,1,1]
				for i in range(20):
					for j in range(8):
						rp = ra.choice(p)
						rl.append(rp)
					rlist.append(rl)
					rl = []
				pr = [4,4,3,3,3,3,2,2]
				for i in range(10):
					for j in range(8):
						ps = ra.choice(pr)
						rml.append(ps)
					rmlist.append(rml)
					rml = []
				act1=20;act2=70
			if yr == 1 and q == 4:
				rlist = []
				rmlist = []
				p = [np.nan,4,4,4,3,3,3,2,2,1,2]
				for i in range(20):
					for j in range(8):
						rp = ra.choice(p)
						rl.append(rp)
					rlist.append(rl)
					rl = []
				pr = [4,4,4,3,3,3,2,2]
				for i in range(10):
					for j in range(8):
						ps = ra.choice(pr)
						rml.append(ps)
					rmlist.append(rml)
					rml = []
				act1=30;act2=80
			if yr == 2 and q == 2:
				rlist = []
				rmlist = []
				p = [np.nan,4,4,4,3,3,3,3,2,2,2]
				for i in range(20):
					for j in range(8):
						rp = ra.choice(p)
						rl.append(rp)
					rlist.append(rl)
					rl = []
				pr = [5,4,4,3,3,3,2,2]
				for i in range(10):
					for j in range(8):
						ps = ra.choice(pr)
						rml.append(ps)
					rmlist.append(rml)
					rml = []
				act1=35;act2=80
			if yr == 2 and q == 2:
				rlist = []
				rmlist = []
				p = [np.nan,5,4,4,4,3,3,3,3,2,2,2]
				for i in range(20):
					for j in range(8):
						rp = ra.choice(p)
						rl.append(rp)
					rlist.append(rl)
					rl = []
				pr = [5,5,4,4,3,3,3,2,2]
				for i in range(10):
					for j in range(8):
						ps = ra.choice(pr)
						rml.append(ps)
					rmlist.append(rml)
					rml = []
				act1=40;act2=85
			if yr == 2 and q == 3:
				rlist = []
				rmlist = []
				p = [np.nan,5,4,4,4,3,3,3,2,2]
				for i in range(20):
					for j in range(8):
						rp = ra.choice(p)
						rl.append(rp)
					rlist.append(rl)
					rl = []
				pr = [5,5,4,4,3,3,2]
				for i in range(10):
					for j in range(8):
						ps = ra.choice(pr)
						rml.append(ps)
					rmlist.append(rml)
					rml = []
				act1=45;act2=90
			if yr == 2 and q == 4:
				rlist = []
				rmlist = []
				p = [np.nan,5,5,4,4,4,3,3,3,2]
				for i in range(20):
					for j in range(8):
						rp = ra.choice(p)
						rl.append(rp)
					rlist.append(rl)
					rl = []
				pr = [5,5,4,4,3,3,2]
				for i in range(10):
					for j in range(8):
						ps = ra.choice(pr)
						rml.append(ps)
					rmlist.append(rml)
					rml = []
				act1=50;act2=100
			if yr == 3 and q == 1:
				rlist = []
				rmlist = []
				p = [np.nan,5,5,5,4,4,4,3,3,3,2]
				for i in range(20):
					for j in range(8):
						rp = ra.choice(p)
						rl.append(rp)
					rlist.append(rl)
					rl = []
				pr = [5,5,4,4,3,3]
				for i in range(10):
					for j in range(8):
						ps = ra.choice(pr)
						rml.append(ps)
					rmlist.append(rml)
					rml = []
				act1=50;act2=100
			if yr == 3 and q == 2:
				rlist = []
				rmlist = []
				p = [np.nan,5,5,5,4,4,4,4,3,3,2]
				for i in range(20):
					for j in range(8):
						rp = ra.choice(p)
						rl.append(rp)
					rlist.append(rl)
					rl = []
				pr = [5,5,4,4,4,3]
				for i in range(10):
					for j in range(8):
						ps = ra.choice(pr)
						rml.append(ps)
					rmlist.append(rml)
					rml = []
				act1=65;act2=115
			if yr == 3 and q == 3:
				rlist = []
				rmlist = []
				p = [np.nan,5,5,5,5,4,4,4,4,3,3,2]
				for i in range(20):
					for j in range(8):
						rp = ra.choice(p)
						rl.append(rp)
					rlist.append(rl)
					rl = []
				pr = [5,5,5,4,4,4,3]
				for i in range(10):
					for j in range(8):
						ps = ra.choice(pr)
						rml.append(ps)
					rmlist.append(rml)
					rml = []
				act1=75;act2=125
			if yr == 3 and q == 4:
				rlist = []
				rmlist = []
				p = [np.nan,5,5,5,5,5,4,4,4,4,3,3,2]
				for i in range(20):
					for j in range(8):
						rp = ra.choice(p)
						rl.append(rp)
					rlist.append(rl)
					rl = []
				pr = [5,5,5,4,4,4,3,5]
				for i in range(10):
					for j in range(8):
						ps = ra.choice(pr)
						rml.append(ps)
					rmlist.append(rml)
					rml = []
				act1=75;act2=135
			part_regis(n, q, yr)
			mgt_ratings(n, q, yr, rmlist)
			part_web_act(n, q, yr, act1, act2)
			part_satisfaction(n, r, yr, q, rlist)
			part_web_sat(n, r, yr, q, rlist)
			part_team_sat(n, r, yr, q, rlist)
