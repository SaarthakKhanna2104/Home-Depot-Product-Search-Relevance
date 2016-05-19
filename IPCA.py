from sklearn.decomposition import IncrementalPCA
#import pandas as pd
import numpy as np
import generate_features as gf
import time


def ipca():
	train_features, test_features = gf.get_tfidf()
	vectorizer = gf.get_tfidf()
	n_components = 250
	ipca = IncrementalPCA(n_components=n_components, batch_size=1250)
	start_time = time.time()
	print 'start ipca on train'
	X_ipca = ipca.fit_transform(train_features)
	runtime = time.time() - start_time
	print '-----'
	print '%.2f seconds to ipca on train' % runtime
	print '-----'
	train_features = None
	
	print 'ipca train done'
	np.savetxt('train_features.csv', X_ipca, fmt='%.8e', delimiter=",")
	X_ipca = None
	print 'ipca train file done'
	test_features = gf.get_tfidf(vectorizer, False)
	Y_ipca = ipca.fit_transform(test_features)
	test_features, vectorizer = None, None
	print 'ipca test done'
	np.savetxt('test_features.csv', Y_ipca, fmt='%.8e', delimiter=",")
	svd_test_features = None
	print 'ipca test file done'

if __name__ == '__main__':
	ipca()