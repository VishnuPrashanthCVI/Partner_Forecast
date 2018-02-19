import pandas as pd
import numpy as np
import pickle as pkl

def load_data():
	with open('1_to_300_raw_partner_data.pkl', 'rb') as f:
		data_norm = pkl.load(f)

	with open('301_to_400_raw_partner_data.pkl', 'rb') as f:
		data_up = pkl.load(f)

		with open('401_to_500_raw_partner_data.pkl', 'rb') as f:
			data_down = pkl.load(f)

	raw_partner_data = pd.concat([data_down,data_norm,data_up], ignore_index = True)
	raw_partner_data.sort_values(by = ['ID','Yr','Qtr'], axis = 0)
	raw_partner_data.reset_index(drop = True)

	filestr = 'raw_partner_data.pkl'
	with open(filestr,'wb') as f:
		pkl.dump(raw_partner_data,f,-1)

	with open('1_to_300_true_partner_data.pkl', 'rb') as f:
		data_norm = pkl.load(f)

	with open('301_to_400_true_partner_data.pkl', 'rb') as f:
		data_up = pkl.load(f)

	with open('401_to_500_true_partner_data.pkl', 'rb') as f:
		data_down = pkl.load(f)

	true_partner_data = pd.concat([data_down,data_norm,data_up], ignore_index = True)

	true_partner_data.sort_values(by = ['ID','Yr','Qtr'], axis = 0)
	true_partner_data.reset_index(drop = True)

	filestr = 'true_partner_data.pkl'
	with open(filestr,'wb') as f:
		pkl.dump(true_partner_data,f,-1)

	return

if __name__== '__main__':

	load_data()
