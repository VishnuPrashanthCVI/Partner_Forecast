import pandas as pd
import numpy as np
import random
from sklearn.utils import resample
import pickle as pkl

#number of Partners
n = 300 #original sample
r = 100 #resample size
rlist = [np.nan,5,5,5,4,4,4,3,2,np.nan]
#create partner index
ds = pd.DataFrame()
ds['ID'] = range(1,n + 1)

#create partner certification
clist=['Platinum','Gold','Silver','Premier', 'Registered']
Cert = []
for i in range(n):
	cert = random.choice(clist)
	Cert.append(cert)

#create partner primary channel
dlist=['VAR', 'DVAR', 'DIST']
RTM = []
for i in range(n):
	rtm = random.choice(dlist)
	RTM.append(rtm)

#create primary customer target
pclist=['ENT', 'ENT', 'CAR', 'CAR', 'EDUC', 'GOVT', 'ENT']
Cust = []
for i in range(n):
	cust = random.choice(pclist)
	Cust.append(cust)

#create credit rating
credit = range(4,11)
Credit = []
for i in range(n):
	cr = random.choice(credit)
	Credit.append(cr)

#create customer target size
cslist=['10','100','250','1000','10000','25000']
Cust_Size = []
for i in range(n):
	cust = random.choice(cslist)
	Cust_Size.append(cust)

#create territory
tlist=['NE', 'SE', 'SW', 'NW', 'C']
Terr = []
for i in range(n):
	terr = random.choice(tlist)
	Terr.append(terr)

#create basic data file
ds['Cert'] = Cert
ds['RTM'] = RTM
ds['Cust'] = Cust
ds['Cust_Size'] = Cust_Size
ds['Credit'] = Credit
ds['Territory'] = Terr

#create multiple respondents of exec, sales, operations, staff
#note resample is a numpy utility that works for pandas also
dss = resample(ds,n_samples=r,random_state=71)
ds = pd.concat([ds,dss], axis = 0, ignore_index = True)
slist=['Exec', 'Exec', 'Sales','Sales','Sales','Opns','Staff']
Resp = []
for i in range(n+r):
	resp = random.choice(slist)
	Resp.append(resp)
ds['Resp'] = Resp
ds.sort_values(by = 'ID', ascending=True, inplace=True)
Resp = []

#create general satisfaction questions
questions = ['Satisfaction', 'Value_Potential', 'Expected_Relationship_Duration', 'Profit_Potential', 'Product_Quality', 'Product_Importance', 'Product_Breadth', 'Product_Competitiveness', 'Product_Training', 'Customer_Referral', 'Profit_Expectation', 'Effective_Communications', 'Price_Point', 'Margin', 'Brand_Requirement', 'Training_Effectiveness', 'Customer_Recognition', 'Problem_Solving']
#gresp = np.zeros((n+r,len(questions)))


for col in questions:
	Resp = []
	for i in range(n+r):
		resp = random.choice(rlist)
		Resp.append(resp)
	ds[col] = Resp

f = open('partner_sat_data_3_1.pkl', 'wb')
pkl.dump(ds,f,-1)
f.close()

#create account team ratings
questions = []
Resp = []
ds = pd.DataFrame()
ds['ID'] = range(1,n + 1)
questions = ['Acct_Mgr', 'Support_Mgr', 'Team_Ability', 'Team_Contribution', 'Operations_Ability', 'Acct_Team_Satisfaction', 'Team_Solutions', 'Overall_Satisfaction']
for col in questions:
	Resp = []
	for i in range(n):
		resp = random.choice(rlist)
		Resp.append(resp)
	ds[col] = Resp
f = open('partner_acct_data_3_1.pkl', 'wb')
pkl.dump(ds,f,-1)
f.close()

#create online ratings and usage data
questions = []
Resp = []
ds = pd.DataFrame()
ds['ID'] = range(1,n+1)
questions = ['Usefulness', 'Effectiveness', 'Response', 'Error_Free', 'Ease_of_Use', 'Pricing', 'Delivery_Response', 'Overall_Satisfaction']

for col in questions:
	Resp = []
	for i in range(n):
		resp = random.choice(rlist)
		Resp.append(resp)
	ds[col] = Resp
dss = resample(ds,n_samples=r,random_state=71)
ds = pd.concat([ds,dss], axis = 0, ignore_index = True)
slist=['Engr', 'Sales', 'Sales']
Resp = []
for i in range(n+r):
	resp = random.choice(slist)
	Resp.append(resp)
ds['Resp'] = Resp
ds.sort_values(by = 'ID', ascending=True, inplace=True)
questions = ['Usefulness', 'Effectiveness', 'Response', 'Error_Free', 'Ease_of_Use', 'Pricing', 'Delivery_Response', 'Overall_Satisfaction']
for col in questions:
	Resp = []
	for i in range(ds.shape[0]):
		resp = random.choice(rlist)
		Resp.append(resp)
	ds[col] = Resp

f = open('partner_web_rating_3_1.pkl', 'wb')
pkl.dump(ds,f,-1)
f.close()

#create company rankings of partner
ds = pd.DataFrame()
ds['ID'] = range(1,n+1)
cols = ['Sales','Engr','Training','Support','Operations']
for col in cols:
	Resp = []
	for i in range(ds.shape[0]):
		resp = random.choice(rlist)
		Resp.append(resp)
	ds[col] = Resp

f = open('partner_rating_3_1.pkl', 'wb')
pkl.dump(ds,f,-1)
f.close()
