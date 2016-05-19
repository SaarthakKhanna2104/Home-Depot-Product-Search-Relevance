import json
import pandas as pd
import sys
import re
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.stem.porter import *
import time
import symSpell

dictionary = {}


def get_json_df(inputjson):
	inputdata = open(inputjson)
	df = pd.read_json(inputdata)
	return df


def clean_data(data, spellCheck = False):
	letters_digits = re.sub("[^a-zA-Z0-9]", " ", data)
	words = letters_digits.lower().split()
	stops = set(stopwords.words("english"))
	if spellCheck:
		meaningful_words = [PorterStemmer().stem(symSpell.get_suggestions(word)) if word.isalpha() else  PorterStemmer().stem(word) for word in words if not word in stops]
	else:
		meaningful_words = [PorterStemmer().stem(w) for w in words if not w in stops]
	#meaningful_words = [PorterStemmer().stem(w) for w in words if not w in stops]
	return( " ".join( meaningful_words ))


def processData(inputjson, isTrain = True):
	df = get_json_df(inputjson)
	search_list = list()
	start_time = time.time()
	symSpell.create_dictionary("big.txt")
	runtime = time.time() - start_time
	print '-----'
	print '%.2f seconds to create dictionary' % runtime
	print '-----'

	for index, row in df.iterrows():
		
		query = row['title'] + ' ' + row['description']

		for attr in df.loc[index, 'attributes']:
			query = query + ' ' + attr['name']+ ' ' + attr['value']

		for s in df.loc[index, 'search']:
			
			search_dict = {}
			search_dict['id'] = s['id']
			
			search_dict['query'] = clean_data(query, True) + ' ' + clean_data(s['search_term'])
			if isTrain:
				search_dict['relevance'] = s['relevance']
			search_list.append(search_dict)

	filename = 'processed_training_data.json' if isTrain else 'processed_test_data.json'
	output_file = open(filename,"w+")
	json.dump(search_list,output_file,sort_keys=True,indent=4,encoding='latin1')
	output_file.close()


if __name__=='__main__':
	inputjson = sys.argv[1]
	start_time = time.time()
	processData(inputjson)
	runtime = time.time() - start_time
	print '-----'
	print '%.2f seconds to run' % runtime
	print '-----'
	print 'done'