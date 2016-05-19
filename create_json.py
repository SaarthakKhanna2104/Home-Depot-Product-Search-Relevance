import pandas as pd
import json
import re
import time
import json_parser
import math


def get_json_data(prod_desc_df, train_df, attribute_df, isTrain = True):
	product_list = list()

	jsondata = open('spell_check.json')
	spell_check_dict = json_parser.json_load_byteified(jsondata)
	spellSet = set(spell_check_dict.keys())

	for desc_index,desc_row in prod_desc_df.iterrows():
		i=i+1
		prod_dict = {}
		attributes_list = []
		search_list = []
		temp_train_df = pd.DataFrame()

		prod_dict['product_uid'] = desc_row['product_uid']
		prod_dict['description'] = desc_row['product_description']
		temp_attr_df = attribute_df.loc[attribute_df['product_uid'] == desc_row['product_uid']]
		
		for attr_index,attr_row in temp_attr_df.iterrows():
			attr_dict = {}
			attr_dict['name'] = attr_row['name']
			attr_dict['value'] = attr_row['value']#if math.isnan(float(attr_row['value'])) else "None"
			attributes_list.append(attr_dict)
		prod_dict['attributes'] = attributes_list
		
		temp_train_df = train_df.loc[train_df['product_uid'] == desc_row['product_uid']]
		if not temp_train_df.empty:
			prod_dict['title'] = temp_train_df.iloc[0]['product_title']
			for train_index,train_row in temp_train_df.iterrows():
				train_dict = {}
				train_dict['id'] = train_row['id']
				search_term = train_row['search_term']
				train_dict['search_term'] = spell_check_dict[search_term] if search_term in spellSet else search_term
				if isTrain:
					train_dict['relevance'] = train_row['relevance']
				search_list.append(train_dict)

			prod_dict['search'] = search_list
			product_list.append(prod_dict)
		
		if len(product_list)%1000 == 0:
			print "%d rows added"%(len(product_list))

	filename = "training_data.json" if isTrain else "test_data.json"
	output_file = open(filename, "w+")
	json.dump(product_list,output_file,sort_keys=True,indent=4,encoding='latin1')
	output_file.close()