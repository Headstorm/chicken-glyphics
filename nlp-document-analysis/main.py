## Import Libraries
import textExtraction as te
import importUtils
from tqdm import tqdm
import pandas as pd
from os import listdir
from os.path import isfile, join
import plotUtils as plotu
import mlUtils as ml
import numpy

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

MAX_READ_RESUME = -1
NUM_TOP_USED_WORD = 100
MIN_COUNT = 5
PCA_COMP = 8
K_CLUS = 2

if __name__ == '__main__':

	print("----Load Resume----")
	resume_name_list = importUtils.getListOfFiles("Original_Resumes")
	valid_file_list, resume_string_extract = importUtils.read_files_tika(resume_name_list, max_read = MAX_READ_RESUME)

	dataset = pd.DataFrame(columns=['file_name','raw_text','emails','phone_numbers','names','gpas','tokens_nltk', 'tokens_keras', 'tokens_id_encoded'])
	# dataset['file_name'] = resume_name_list
	if MAX_READ_RESUME > 0:
		dataset = dataset[0:MAX_READ_RESUME]

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
		dataset.loc[count_text,'file_name'] = valid_file_list[count_text]
		# print("")
		dataset.loc[count_text,'raw_text'] = resume_text
		# print("-")
		dataset.loc[count_text,'tokens_nltk'] = words
		# print("--")
		dataset.loc[count_text,'emails'] = emails
		# print("---")
		dataset.loc[count_text,'phone_numbers'] = numbers
		# print("----")
		dataset.loc[count_text,'names'] = names
		# print("-----")
		dataset.loc[count_text,'gpas'] = gpas
		# print("------")

		count_text = count_text + 1

	dataset = dataset.dropna(subset=['raw_text'])
	## Encode text with keras
	# dataset['tokens_keras'] = te.extract_token_keras(dataset['raw_text'], NUM_TOP_USED_WORD)

	## Merge All documents tokens to one long list
	all_resumes_nltk_tokens = te.merge_all_documents_token_to_one(dataset['tokens_nltk'])

	## Visualized most used N words
	dict_word_top_used = te.get_words_df_by_top_used(all_resumes_nltk_tokens, NUM_TOP_USED_WORD)
	plotu.plot_bar(dict_word_top_used['word'], dict_word_top_used['frequency'],
	 'Word', 'Frequency','top_'+str(NUM_TOP_USED_WORD)+'_word')

	#### Word Encoding
	## Get the word dictionary with all words that appear more than 5 times
	dict_word_used_at_least_min = te.get_words_df_by_at_least_min_count(all_resumes_nltk_tokens, MIN_COUNT)
	
	## Encode words with it's ID from previous word dictionary
	for i in tqdm(range(len(dataset)), desc="Encode word wtih dict"):
		dataset.loc[i, 'tokens_id_encoded'] = te.encode_word_with_dict(dataset.loc[i, 'tokens_nltk'], dict_word_used_at_least_min)
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


