import pandas as pd
import numpy as np
import random as ra
import pickle as pkl

ra.seed(73)

#creating a data file with all id and multiple respondents per id
#create partner index including all respondents
def create_respondents(YR=[1,2,3,4],Q=[1,2,3,4],n=300):
	partner_id=list(range(1,n+1))
	no = [3,2,3,2,1.1]
	ID = []
	Yr = []
	Qtr = []
	for id in partner_id:
		for yr in YR:
			for qtr in Q:
				resp = ra.choice(no)
				for l in range(int(resp)):
					ID.append(id)
					Qtr.append(qtr)
					Yr.append(yr)
	return partner_id,ID,Yr,Qtr

#create a dictionary of partner_id and registration for all respondents
def create_identity_dict(partner_id):
	iden_dict={}
	clist=['Platinum','Gold','Silver','Premier', 'Registered','Gold','Silver','Premier', 'Registered','Registered']
	dlist=['VAR', 'DVAR', 'DIST']
	plist=['ENT', 'ENT', 'CAR', 'CAR', 'ENT', 'GOVT', 'ENT', 'EDUC']
	cslist=['10','100','250','1000','10000','25000','100','250','1000','10000']
	tlist=['NE', 'SE', 'SW', 'NW', 'C', 'W', 'NE']
	crlist = [5,5,5,4,4,4,3,3,2,1]

	for id in partner_id:
		c = ra.choice(clist)
		d = ra.choice(dlist)
		p = ra.choice(plist)
		ca = ra.choice(cslist)
		t = ra.choice(tlist)
		cr =  ra.choice(crlist)
		iden_dict[id]=[c,d,p,ca,t,cr]

	return iden_dict

#create partner respondents registration data pd file
def create_response_df(ID,Yr,Qtr,iden_dict):
	df=pd.DataFrame()
	class_list = []
	type_list = []
	customer_type = []
	customer_size = []
	territory = []
	credit = []
	respondent =[]


	resplist=['Mgt','Sales','Engr','Opn', 'Support', 'Mgt','Mgt','Mgt']
	for id in ID:
		class_list.append(iden_dict[id][0])
		type_list.append(iden_dict[id][1])
		customer_type.append(iden_dict[id][2])
		customer_size.append(iden_dict[id][3])
		territory.append(iden_dict[id][4])
		credit.append(iden_dict[id][5])
		respondent.append(ra.choice(resplist))
	df['Response_Number']=range(len(ID))
	df['ID']=ID
	df['Yr']=Yr
	df['Qtr']=Qtr
	df['Class']=class_list
	df['Partner_Type']=type_list
	df['Customer_Type'] = customer_type
	df['Customer_Size'] = customer_size
	df['Territory'] = territory
	df['Credit_Rating'] = credit
	df['Respondent'] = respondent
	df.sort_values(by = ['Response_Number'], ascending=True, inplace=True)
	return df


def make_scores_dict(df):
	time_list=[]
	for i in range(1,5):
		for j in range(1,5):
			time_list.append((i,j))

	score = []
	score.append([3,2,2,2,1,np.nan,np.nan,1,1,2,2,2,3])
	score.append([3,3,2,2,1,np.nan,np.nan,1,1,2,2,3,3])
	score.append([3,3,3,2,1,np.nan,np.nan,1,1,2,2,3,4])
	score.append([3,3,3,2,2,np.nan,np.nan,1,2,2,2,3,4])
	score.append([4,3,3,2,2,np.nan,np.nan,2,2,2,3,3,4])
	score.append([4,3,3,2,2,np.nan,np.nan,2,2,2,2,3,4])
	score.append([4,3,3,3,2,np.nan,np.nan,2,2,2,3,3,4])
	score.append([4,4,3,3,2,np.nan,np.nan,2,2,3,3,3,4])
	score.append([4,4,3,3,2,np.nan,np.nan,2,2,3,3,4,4])
	score.append([4,4,3,3,3,np.nan,np.nan,2,2,3,3,4,4])
	score.append([4,4,4,3,3,np.nan,np.nan,2,2,3,4,4,4])
	score.append([5,4,4,3,3,np.nan,np.nan,3,3,3,4,4,4])
	score.append([5,4,4,3,3,np.nan,np.nan,3,3,4,4,4,5])
	score.append([5,5,4,3,3,np.nan,np.nan,3,3,4,4,4,5])
	score.append([5,5,4,4,3,np.nan,np.nan,3,3,4,4,5,5])
	score.append([5,5,4,4,4,np.nan,np.nan,3,4,4,5,5,5])
	scores = dict(zip(time_list,score))
	return scores

def make_entry(df,scores):
	satisfaction = ['Overall_Satisfaction', 'Value_Potential', 'Expected_Relationship_Duration', 'Profit_Potential', 'Product_Quality', 'Product_Importance', 'Product_Breadth', 'Product_Competitiveness', 'Product_Training', 'Customer_Referral', 'Profit_Expectation', 'Effective_Communications', 'Price_Point', 'Margin', 'Brand_Requirement', 'Training_Effectiveness', 'Customer_Recognition', 'Problem_Solving']
	mgt = ['Sales','Engr','Training','Support','Operations','Expert']
	activity = ['Bids','Quotes','Registrations','Inquiries','Sessions']
	acct_team = ['Acct_Mgr', 'Support_Mgr', 'Team_Ability', 'Team_Contribution', 'Operations_Ability', 'Acct_Team_Satisfaction', 'Team_Solutions', 'Overall_Satisfaction']
	sat_web = ['Usefulness', 'Effectiveness', 'Response', 'Error_Free', 'Ease_of_Use', 'Pricing', 'Delivery_Response', 'Overall_Satisfaction']
	features = satisfaction+mgt+activity+acct_team+sat_web

	idt=[]
	idt = zip(list(df.Yr),list(df.Qtr))
	for col in features:
		T=[]
		for i in range(len(idt)):
			T.append(ra.choice(scores[idt[i]]))
		df[col]=T
	df = df.sort_values(by = ['Response_Number'], ascending=True, inplace=False).reset_index(drop=True)
	return df

if __name__== '__main__':

	partner_id,ID,Yr,Qtr=create_respondents(YR=[1,2,3,4],Q=[1,2,3,4],n=300)
	iden_dict=create_identity_dict(partner_id)
	df=create_response_df(ID,Yr,Qtr,iden_dict)
	scores=make_scores_dict(df)
	df = make_entry(df,scores)
	filestr = 'raw_partner_data.pkl'
	with open(filestr,'wb') as f:
		pkl.dump(df,f,-1)
