from sklearn import svm
from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np
import pickle

def get_json_df(inputjson):
	inputdata = open(inputjson)
	df = pd.read_json(inputdata)
	return df

def get_target_array():
	df = get_json_df('processed_training_data.json')
	return np.array(df['relevance'])

def svrModel(kern, train_features, targets):
	print len(targets)
	print len(train_features[0])
	#print len(test_features)
	regr_svr = svm.SVR()

	C = [1, 10]
	gamma = [0.001, 0.002, 0.003] #  0.005, 0.004, 
	epsilon=[0.01]
	kernel = list(kern)
	degree = [2, 3, 4]

	parameters = {"C":C, "gamma":gamma, "epsilon":epsilon,  "kernel":kernel, "degree":degree}
	gs = GridSearchCV(regr_svr, parameters, scoring="mean_squared_error", n_jobs = 2, cv=3, verbose=10)
	gs.fit(train_features, targets)

	print "Best Estimator:\n%s"  % gs.best_estimator_
	print "Scoring:\n%s" % gs.scorer_
	print "Scoring:\n%f" % gs.best_score_
	final_svr = gs.best_estimator_
	return final_svr


if __name__=='__main__':
	train_features = np.genfromtxt('train_features.csv', delimiter=",")
	targets = get_target_array()
	kernels = ['rbf', 'linear', 'poly']
	filenames = ['SVR_model_rbf.pkl', 'SVR_model_linear.pkl', 'SVR_model_poly.pkl']
	for kernel in kernels:
		svrModel(kernel, train_features, targets)
		ind = kernels.index(kernel)
		filename = filenames[ind]
		pickle.dump( final_svr, open(filename, 'wb' ))