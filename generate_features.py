import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time

def get_json_df(inputjson):
	inputdata = open(inputjson)
	df = pd.read_json(inputdata)
	return df

def get_tfidf(vec = None, isTrain = True):
	if isTrain:
		train_df = get_json_df('processed_training_data.json')
		sentences = list(train_df['query'])
		vectorizer = TfidfVectorizer(min_df=1) #Convert a collection of raw documents to a matrix of TF-IDF features.
		X_array = vectorizer.fit_transform(sentences) #Learn the vocabulary dictionary and return term-document matrix.
		train_df, sentences = None, None
		X_array = X_array.todense()
		print 'features built'
		return vectorizer, X_array

	else:
		test_df = get_json_df('processed_test_data.json')
		test_sentences = list(test_df['query'])
		print 'transforming'
		start_time = time.time()
		Y_array = vec.transform(test_sentences)
		runtime = time.time() - start_time
		print '-----'
		print '%.2f seconds to run' % runtime
		print '-----'
		print 'shit is transformed'
		test_df, test_sentences, vec = None, None, None
		Y_array = Y_array.todense()
		print len(Y_array)
		print len(Y_array[0])
		print 'test features built'
		return Y_array


if __name__ == '__main__':
	get_tfidf()