import sys
import os.path
import pandas as pd
import numpy as np
import json
import time
import create_json
import preProcessing
import IPCA
import Predictor

def get_json_df(inputjson):
	inputdata = open(inputjson)
	df = pd.read_json(inputdata)
	return df


def create_target_array():
	df = get_json_df('processed_training_data.json')
	id_array = np.array(df['id'])
	np.savetxt('ids.csv', id_array, fmt='%d', delimiter=",")
	rel_array = np.array(df['relevance'])
	np.savetxt('targets.csv', rel_array, fmt='%.2f', delimiter=",")


def get_df(file):
	df = pd.read_csv(file,sep=',',header=0)
	return df


def get_training_processed(prod_desc_df,train_df,attribute_df):
	#Convert the input data into JSON format
	create_json.get_json_data(prod_desc_df,train_df,attribute_df)

	#Perform pre-processing on the JSON input data
	preProcessing.processData('training_data.json')


def get_test_processed(prod_desc_df,test_df,attribute_df):
	#Convert the test data into JSON format
	create_json.get_json_data(prod_desc_df,test_df,attribute_df, False)
	#Perform pre-processing on the JSON input data
	preProcessing.processData('test_data.json', False)


if __name__ == '__main__':
	prod_desc_file = sys.argv[1]
	train_file = sys.argv[2]
	attribute_file = sys.argv[3]
	
	#Create dataframes for the input files
	prod_desc_df = get_df(prod_desc_file)
	train_df = get_df(train_file)
	attribute_df = get_df(attribute_file)

	test_df = get_df('test.csv')
	

	start_time = time.time()

	if not os.path.isfile('train_features.csv') and not os.path.isfile('test_features_test.csv'):
		get_training_processed(prod_desc_df,train_df,attribute_df)
		get_test_processed(prod_desc_df,test_df,attribute_df)

		#Generate features
		IPCA.ipca()

	#Make Predictions and get root mean squared error for each model
	if os.path.isfile('SVR_model_rbf.pkl') and os.path.isfile('SVR_model_linear.pkl') and os.path.isfile('SVR_model_poly.pkl') and os.path.isfile('GBR_model.pkl') and os.path.isfile('random_forest_model.pkl'):
		error_dict = Predictor.get_predictions()
		print error_dict

	runtime = time.time() - start_time
	print '-----'
	print '%.2f seconds to run' % runtime
	print '-----'
	print 'done'