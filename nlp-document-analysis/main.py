from os import listdir
from os.path import isfile, join
import os
import glob
import textract
from tqdm import tqdm
import docx2txt
import tika
from tika import parser
import pandas as pd
import textExtraction as te
import importUtils
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import plotUtils as plotu
import mlUtils as ml
import numpy as np


MAX_READ_RESUME = -1
NUM_TOP_USED_WORD = 100
MIN_COUNT = 5
PCA_COMP = 8
K_CLUS = 2

if __name__ == '__main__':

	print("----Load Resume----")
	resume_name_list = importUtils.getListOfFiles("greenhouse-resumes")

	data_rejected = pd.read_csv("candidates_rejected.csv")
	data_hired = pd.read_csv("candidates_hired.csv")

	dataset = pd.concat([data_hired, data_rejected])

	FILE_NAME_PREFIX = "greenhouse-resumes\\"

	dataset['filename'] = FILE_NAME_PREFIX + dataset['First Name'] + " " + dataset['Last Name'] + ".pdf"
	dataset['filename'] = dataset['filename'].str.replace('"','')

	got_filename_from_sheet = list(dataset['filename'])

	count_found = 0

	found_resume = list()

	for resume_name in resume_name_list:
		if resume_name in got_filename_from_sheet:
			count_found = count_found + 1
			found_resume.append(resume_name)
		else:
			print("Can't find : ", resume_name)

	print("Resume Found : ", count_found)

	dataset = dataset[dataset.filename.isin(found_resume)]
	dataset['Application Date'] = pd.to_datetime(dataset['Application Date'])
	dataset = dataset.sort_values('Application Date').drop_duplicates(['filename'] ,keep='last')
	dataset.reset_index(drop=True, inplace=True)

	valid_file_list, resume_string_extract = importUtils.read_files_tika(dataset['filename'], max_read = MAX_READ_RESUME)

	dataset["tokens_nltk"] = np.nan
	dataset["tokens_nltk"] = dataset["tokens_nltk"].astype('object')
	dataset["emails"] = np.nan
	dataset["emails"] = dataset["emails"].astype('object')
	dataset["names"] = np.nan
	dataset["names"] = dataset["emails"].astype('object')
	dataset["phone_numbers"] = np.nan
	dataset["phone_numbers"] = dataset["phone_numbers"].astype('object')
	dataset["gpas"] = np.nan
	dataset["gpas"] = dataset["gpas"].astype('object')

	# dataset['text'] = resume_string_extract
	print("\n----Clean text and Extract A lot of stuff----")
	count_text = 0
	for resume_text in tqdm(resume_string_extract, desc="Clean and extract text"):
		# print(resume_text)
		##Convert Byte to String
		# print(count_text, "-Decode")
		# resume_text = resume_text.decode("utf-8")
		
		##Tokenize each resume
		# print(count_text, "-NLTK Tokenize")
		tokens = te.extract_casual_to_tokens(resume_text)
		words = te.remove_word_with_nonalpha(tokens)
		words = te.remove_stopwords(words)
		words = te.remove_single_character(words)
		words = te.remove_months_abbrev(words)
		words = te.stemming_text(words)
		# words = te.lemma_text(words)

		##Extract Email
		# print(count_text, "-Extract Email")
		emails = te.extract_email_addresses(resume_text)

		##Extract Phone Number
		# print(count_text, "-Extract Phone")
		numbers = te.extract_phone_numbers(resume_text)

		##Extract Name
		# print(count_text, "-Extract Name")
		names = te.extract_names(resume_text)

		##Extract GPA
		# print(count_text, "-Extract GPAS")
		gpas = te.extract_gpas(resume_text)

		##Insert to DataFrame
		# print(count_text, "-Add data to dataframe")
		# print(valid_file_list[count_text])
		dataset.loc[count_text,'file_name'] = valid_file_list[count_text]
		# print("")
		# print(resume_text)
		dataset.loc[count_text,'raw_text'] = resume_text
		# print("-")
		# print(words)
		dataset.at[count_text,'tokens_nltk'] = words
		# print("--")
		dataset.at[count_text,'emails'] = emails
		# print("---")
		dataset.at[count_text,'phone_numbers'] = numbers
		# print("----")
		dataset.at[count_text,'names'] = names
		# print("-----")
		dataset.at[count_text,'gpas'] = gpas
		# print("------")
		count_text = count_text + 1

	print(dataset)

	## Merge All documents tokens to one long list
	all_resumes_nltk_tokens = te.merge_all_documents_token_to_one(dataset['tokens_nltk'])
	## Visualized most used N words
	dict_word_top_used = te.get_words_df_by_top_used(all_resumes_nltk_tokens, NUM_TOP_USED_WORD)
	plotu.plot_word_cloud(' '.join(all_resumes_nltk_tokens), "word_cloud_top_"+str(NUM_TOP_USED_WORD))
	plotu.plot_bar(dict_word_top_used['word'], dict_word_top_used['frequency'],
	 'Word', 'Frequency','top_'+str(NUM_TOP_USED_WORD)+'_word')

	#### Word Encoding
	## Get the word dictionary with all words that appear more than 5 times
	dict_word_used_at_least_min = te.get_words_df_by_at_least_min_count(all_resumes_nltk_tokens, MIN_COUNT)
	
	dataset["tokens_id_encoded"] = np.nan
	dataset["tokens_id_encoded"] = dataset["tokens_id_encoded"].astype('object')

	## Encode words with it's ID from previous word dictionary
	for i in tqdm(range(len(dataset)), desc="Encode word wtih dict"):
		temp = te.encode_word_with_dict(dataset.loc[i, 'tokens_nltk'], dict_word_used_at_least_min)
		dataset.at[i, 'tokens_id_encoded'] = temp
	# print(dataset['tokens_id_encoded'])
	
	## Find the average number of words in the document
	avg_num_word_in_doc = te.find_average_num_word(list(dataset['tokens_id_encoded']))
	
	## Create dataframe from token encoded
	encoded_token = pd.DataFrame()

	for xi in dataset['tokens_id_encoded']:
		encoded_token = encoded_token.append(pd.Series(xi, index=list(range(len(xi)))), ignore_index=True)
	print(encoded_token.shape)

	## Strip the tokens of each document to the avg num word
	encoded_token = te.strip_token_to_length_df(encoded_token, avg_num_word_in_doc)
	print(encoded_token.shape)

	token_pca = ml.pca(encoded_token.T, n_comp=PCA_COMP)
	print(token_pca.shape)
	k_mean = ml.k_mean(token_pca, num_clus=K_CLUS).labels_

	# dataset['token_pca'] = token_pca
	dataset['kmean_label'] = k_mean

	dataset.to_csv('dataset.csv', index=False)

	exit()
