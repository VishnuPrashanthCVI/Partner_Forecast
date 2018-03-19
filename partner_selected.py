import pandas as pd
import numpy as np
import pickle as pkl

with open('Time_Series_Predict.pkl', 'rb') as f:
	data = pkl.load(f)

dS = data[data.Territory == 'NE']
dS = dS[dS.Class == 'Premier']

dI = pd.DataFrame(dS.ID)
dI['Change'] = dS.Change
dI['Date'] = dS.Date

dID = dI[dI.Date == '2014-11-30'].reset_index(drop=True)
dID = dID.sort_values(by=['Change'], ascending=False)
dID = dID[dID.Change > 0]

df = pd.DataFrame()
ids = list(int(id) for id in dID.ID)
for id in ids:
	dt = dS[dS.ID == id]
	df = pd.concat([df,dt],ignore_index=True)

#df.to_csv('Partner_Select_Mon.csv',index=False)
with open('Partner_Select_Mon.pkl','wb')as f:
	pkl.dump(df,f,-1)

data=None
with open('Part_Qtr_Rev_Data.pkl', 'rb') as f:
	data = pkl.load(f)

df,dt = pd.DataFrame(),pd.DataFrame()
for id in ids:
	dt = data[data.ID == id]
	df = pd.concat([df,dt],ignore_index=True)

#df.to_csv('Partner_Select_Qtr.csv',index=False)

with open('Partner_Select_Qtr.pkl','wb') as f:
	pkl.dump(df,f,-1)
