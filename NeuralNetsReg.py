import sys
import pandas as pd
import numpy as np
import pickle

from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.structure.modules import SoftmaxLayer
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet

import matplotlib.pyplot as plt


def predict_output(net,X_test,y_test):
	predictions = list()
	for tup in X_test:
		predictions.append(net.activate(tup))
	preds = np.asarray(predictions)
	mse = np.mean((preds - y_test)**2)
	print "Mean Squared error: ",mse
	return preds



def run_mlp_regressor(ds,hidden_units=50,epochs=10):
	net = buildNetwork(250,hidden_units,1,bias=True,
					hiddenclass=SigmoidLayer,
					outclass=LinearLayer)
	
	trainer = BackpropTrainer(net,ds,verbose=True)
	train_error, cv_error = trainer.trainUntilConvergence(maxEpochs=epochs,verbose=True)

	return net


def get_test_data_features(X):
	tuples_X = [tuple(map(float,tuple(x))) for x in X.values]
	return tuples_X


def get_dataset_for_pybrain_regression(X,y):
	ds = SupervisedDataSet(250,1)
	tuples_X = [tuple(map(float,tuple(x))) for x in X.values]
	tuples_y = [tuple(map(float,(y,))) for y in y.values]
	for X,y in zip(tuples_X,tuples_y):
		ds.addSample(X,y)
	return ds


def get_json_df(inputjson):
 	inputdata = open(inputjson)
 	df = pd.read_json(inputdata)
 	return df

def get_target_array(file_name,flag='Train'):
 	df = get_json_df(file_name)
 	if flag == 'Train':
 		return df['relevance']
 	else:
 		return np.array(df['relevance'])

n_hidden = [30,50,60,100,150,200,300]
best_nn_estimator = None


if __name__ == '__main__':
	train_file_X = sys.argv[1]
	train_file_y = sys.argv[2]
	test_file_X = sys.argv[3]
	test_file_y = sys.argv[4]

	X = pd.read_csv(train_file_X,sep=',')	
	y = get_target_array(train_file_y)
	
	ds = get_dataset_for_pybrain_regression(X,y)

	for h_units in n_hidden:
		best_nn_estimator = run_mlp_regressor(ds,h_units)
	
	pickle.dump(best_nn_estimator,open('nn_model.pkl','wb'))

	X_test = pd.read_csv(test_file_X,sep=',')
	X_test = get_test_data_features(X_test)
	y_test = get_target_array(test_file_y,flag='Test')
	
	preds = predict_output(best_nn_estimator,X_test,y_test)
	np.savetxt('nn_prediction.csv',preds,fmt='%.8f',delimiter=',')

	print 'DONE'