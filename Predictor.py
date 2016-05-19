import pandas as pd
import numpy as np
import pickle
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

def get_json_df(inputjson):
	inputdata = open(inputjson)
	df = pd.read_json(inputdata)
	return df

def get_target_array():
	df = get_json_df('processed_test_data.json')
	return np.array(df['relevance'])

def get_predictions(test_features=None, targets=None):
	pred_dict={}
	if test_features is None and targets is None:
		targets = get_target_array()
		test_features = np.genfromtxt('test_features.csv', delimiter=",")

	svr_model_rbf = pickle.load(open( 'SVR_model_rbf.pkl', 'rb' ))
	rbf_preds = svr_model_rbf.predict(test_features)
	np.savetxt('svr_rbf_predictions.csv', rbf_preds, fmt='%.8f', delimiter=",")
	mse = np.mean((rbf_preds - targets) **2)
	pred_dict['svr_model_rbf'] = mse

	svr_model_linear = pickle.load(open( 'SVR_model_linear.pkl', 'rb' ))
	linear_preds = svr_model_linear.predict(test_features)
	np.savetxt('svr_linear_predictions.csv', linear_preds, fmt='%.8f', delimiter=",")
	mse = np.mean((linear_preds - targets) **2)
	pred_dict['svr_model_linear'] = mse

	svr_model_poly = pickle.load(open( 'SVR_model_poly.pkl', 'rb' ))
	poly_preds = svr_model_poly.predict(test_features)
	np.savetxt('svr_poly_predictions.csv', poly_preds, fmt='%.8f', delimiter=",")
	mse = np.mean((poly_preds - targets) **2)
	pred_dict['svr_model_poly'] = mse

	gbr_model = pickle.load(open( 'GBR_model.pkl', 'rb' ))
	gbr_preds = gbr_model.predict(test_features)
	np.savetxt('gbr_predictions.csv', gbr_preds, fmt='%.8f', delimiter=",")
	mse = np.mean((gbr_preds - targets) **2)
	pred_dict['gbr_model'] = mse

	random_forest_model = pickle.load(open( 'random_forest_model.pkl', 'rb' ))
	rf_preds = random_forest_model.predict(test_features)
	np.savetxt('random_forest_predictions.csv', rf_preds, fmt='%.8f', delimiter=",")
	mse = np.mean((rf_preds - targets) **2)
	pred_dict['random_forest_model'] = mse
	
	return pred_dict



if __name__=='__main__':
	svr_model_rbf = pickle.load(open( 'SVR_model_rbf.pkl', 'rb' ))
	targets = get_target_array()
	test_features = np.genfromtxt('test_features.csv', delimiter=",")
	get_predictions(test_features, targets)
