#Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
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


def GBRModel(X_train,X_cv,y_train,y_cv):
	targets = get_target_array()
	#print len(train_features)
	#print train_features[0]

	#print len(test_features)
	n_estimators = [50, 100]#, 1500, 5000]
	max_depth = [3,8]
	

	best_GBR = None
	best_mse = float('inf')
	best_score = -float('inf')

	print "################# Performing Gradient Boosting Regression ####################### \n\n\n\n"
	for estm in n_estimators:
		for cur_depth in max_depth:
			#random_forest = RandomForestRegressor(n_estimators=estm)
			regr_GBR = GradientBoostingRegressor(n_estimators=estm, max_depth= cur_depth)
			predictor = regr_GBR.fit(X_train,y_train)
			score = regr_GBR.score(X_cv,y_cv)
			mse = np.mean((regr_GBR.predict(X_cv) - y_cv) **2)
			print "Number of estimators used: ",estm
			print "Tree depth used: ",cur_depth
			print "Residual sum of squares: %.2f "%mse
			print "Variance score: %.2f \n"%score
			if best_score <= score:
				if best_mse > mse:
					best_mse = mse
					best_score = score
					best_GBR = predictor	
	print "\nBest score: ",best_score
	print "Best mse: ",best_mse
	return best_GBR
	
	
	
	# parameters = {"n_estimators":n_estimators,"max_depth":max_depth}
	# gs = GridSearchCV(regr_GBR, parameters)
	# gs.fit(train_features, targets)

	# print "Best Estimator:\n%s"  % gs.best_estimator_
	# final_gbr = gs.best_estimator_
	# return final_gbr
	# predicted = final_gbr.predict(test_features)
	# ids = get_test_ids()
	# #index = [i for i in range(len(ids))]
	# columns = ['id']
	# df = pd.DataFrame(data = ids,  columns = columns)#index = index,
	# df['relevance'] = predicted.tolist()
	# print df
	#return df


if __name__=='__main__':
	train_features = np.genfromtxt ('train_features.csv', delimiter=",")
	targets = get_target_array()
	
	X_train,X_cv,y_train,y_cv = train_test_split(train_features,targets,test_size=0.33,random_state=42)
	#svd_features = svd(inputcsv)
	regr_GBR = GBRModel(X_train,X_cv,y_train,y_cv)
	#mse = np.mean((regr_GBR.predict(test_features) - targets) **2)

	#svrPred = svrModel(inputcsv)
	#print svrPred

	pickle.dump(regr_GBR,open('GBR_model.pkl','wb'))
	print 'GBR finitto!!!!!!'
