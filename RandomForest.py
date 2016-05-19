from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

import pandas as pd
import numpy as np
import pickle

def get_json_df(inputjson):
 	inputdata = open(inputjson)
 	df = pd.read_json(inputdata)
 	return df

def get_target_array():
 	df = get_json_df('processed_training_data_subset.json')
 	return np.array(df['relevance'])


def RandomForestModel(X_train,X_cv,y_train,y_cv):
	n_estimators = [5,10,20,30,40,50]

	best_random_forest = None
	best_mse = float('inf')
	best_score = -float('inf')

	print "################# Performing Random Forest ####################### \n\n\n\n"
	for estm in n_estimators:
		random_forest = RandomForestRegressor(n_estimators=estm)
		predictor = random_forest.fit(X_train,y_train)
		score = random_forest.score(X_cv,y_cv)
		mse = np.mean((random_forest.predict(X_cv) - y_cv) **2)
		print "Number of estimators used: ",estm
		print "Residual sum of squares: %.2f "%mse
		print "Variance score: %.2f \n"%score
		if best_score <= score:
			if best_mse > mse:
				best_mse = mse
				best_score = score
				best_random_forest = predictor	

	print "\nBest score: ",best_score
	print "Best mse: ",best_mse
	return best_random_forest




if __name__=='__main__':
	train_features = np.genfromtxt ('train_features.csv', delimiter=",")
	targets = get_target_array()
	
	X_train,X_cv,y_train,y_cv = train_test_split(train_features,targets,test_size=0.33,random_state=42)

	#Get the Random Forest Regression Model
	best_random_forest = RandomForestModel(X_train,X_cv,y_train,y_cv)
	pickle.dump(best_random_forest,open('random_forest_model.pkl','wb'))






