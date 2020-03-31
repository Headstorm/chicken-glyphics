from tika import parser
import pandas as pd
import textract
import nltk
import re
from nltk.tokenize.casual import TweetTokenizer, casual_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import pickle
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

stop = stopwords.words('english')

def merge_all_documents_token_to_one(documents):
	all_resumes_nltk_tokens = []
	for document in documents:
		all_resumes_nltk_tokens = all_resumes_nltk_tokens + document
	return all_resumes_nltk_tokens

def merge_all_ngram_to_one(documents):
	all_trigram_tokens = []
	for document in documents:
		all_trigram_tokens = all_trigram_tokens + document
	return all_trigram_tokens

def get_n_gram_frequency_most_common(ngram_list, n_word):
	fdist = nltk.FreqDist(ngram_list).most_common(n_word)
	return fdist

def get_words_df_by_top_used(documents, num_words):
	fdist = nltk.FreqDist(documents)
	word_common = fdist.most_common(num_words)
	result = pd.DataFrame(columns=['id','word','frequency'])
	# result = result.append({'id': 0, 'word': 'UNK', 'frequency': -1}, ignore_index=True)
	word_id = 1
	for word, frequency in word_common:
		result = result.append({'id': word_id, 'word': word, 'frequency': frequency}, ignore_index=True)
		# print(u'{} : {} : {}'.format(word_id, word, frequency))
		word_id = word_id + 1
	result.to_csv('top_'+str(num_words)+'_words.csv', index=True)
	return result

def get_words_df_by_at_least_min_count(documents, min_count):
	fdist = nltk.FreqDist(documents)
	temp = list(filter(lambda x: x[1]>=min_count,fdist.items()))
	temp.sort(key = lambda x: x[1], reverse=True)
	result = pd.DataFrame(columns=['id','word','frequency'])
	result = result.append({'id': 0, 'word': 'UNK', 'frequency': -1}, ignore_index=True)
	word_id = 1
	for word, frequency in temp:
		result = result.append({'id': word_id, 'word': word, 'frequency': frequency}, ignore_index=True)
		# print(u'{} : {} : {}'.format(word_id, word, frequency))
		word_id = word_id + 1
	result.to_csv('word_dic_with_min_freq_'+str(min_count)+'.csv', index=True)
	return result

def remove_url(document):
	text = re.sub(r'^https?:\/\/.*[\r\n]*', '', document, flags=re.MULTILINE)
	return text

def remove_digits(document):
	result = ''.join([i for i in document if not i.isdigit()])
	return result

def remove_email(document):
	result = ' '.join([i for i in document.split() if '@' not in i])
	return result

def remove_months_abbrev(document):
	result = []
	month_list = ('jan','feb','mar','apr','may',
		'jun','jul','aug','sep','nov','dec')
	for word in document:
		if not word in month_list:
			result.append(word)
	return result

def remove_word_with_nonalpha(document):
	words = set(nltk.corpus.words.words())
	result = [w for w in document if w.lower() in words or not w.isalpha()]
	return result

def remove_stopwords(document):
	result = [w for w in document if not w in stop]
	return result

def extract_casual_to_tokens(document):
	tokens = casual_tokenize(document.lower())
	return tokens

def extract_token_keras(documents, num_words):
	print("----Tokenize using KERAS----")	
	tk = Tokenizer(oov_token='UNK', num_words=num_words+1)
	tk.fit_on_texts(documents)
	# print(tk.word_index)
	# print(tk.texts_to_sequences(documents))
	# tk.word_index = {e:i for e,i in tk.word_index.items() if i <= num_words} # <= because tokenizer is 1 indexed
	sorted_by_word_count = sorted(tk.word_counts.items(), key=lambda kv: kv[1], reverse=True)
	tk.word_index = {}
	i = 0
	for word,count in sorted_by_word_count:
		if i == num_words:
			break
		tk.word_index[word] = i + 1    # <= because tokenizer is 1 indexed
		i += 1
	tk.word_index[tk.oov_token] = num_words + 1
	# saving
	with open('keras_tokenizer.pickle', 'wb') as handle:
		pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)
	result = tk.texts_to_sequences(documents)
	return result

def extract_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', number) for number in phone_numbers]

def extract_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)

def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

def extract_names(document):
    names = []
    sentences = ie_preprocess(document)
    # print(sentences)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
    return names

def extract_sentence(string):
	sent_text = nltk.sent_tokenize(string)
	# print(sent_text)
	return sent_text

def extract_gpas(string):
	r = re.compile(r'([0-9]{1}\.[0-9]{0,2})')
	# gpas = r.findall(string)
	gpa_sent = []
	sent_token = extract_sentence(string)
	# print(sent_token)
	for part in sent_token:
		part = part.lower()
		part_list = part.split()
		# print(part_list)
		if "gpa" in part_list:
			gpa_sent.append(part_list)
	gpa_text = []
	for sent in gpa_sent:
		for word in sent:
			if re.search(r, word):
				gpa_text.append(word)
	gpa_result = []
	for gpa in gpa_text:
		gpa = re.sub(r'[(),]', ' ', gpa)
		gpa = gpa.strip()
		gpa = gpa.split(sep='/')
		gpa_result.append(gpa)
	# print(gpa_result)
	return gpa_result

def remove_slash_tag(document):
	slash_compiled = re.compile(r'^\\[.]*$')
	for word in document:
		print(word)
		if re.search(slash_compiled, word):
			print(word)
	exit()
	resultwords  = [word for word in document if not slash_compiled.match(word)]
	print(resultwords)
	return resultwords

def remove_just_one_symbol(document):
	result = []
	reg = re.compile(r'^.*[-!–$%^&*()_+|~=`’{}\[\]:";\'<>?,.\/]$')
	for word in document:
		# print(word)
		if not re.match(reg, word):
			print(word)

def remove_single_character(document):
	result = []
	for word in document:
		# print(len(word))
		if len(word) > 1:
			result.append(word)
	return result

def remove_number_from_token(document):
	result = []
	reg = re.compile(r'^[0-9]*$')
	for word in document:
		if not re.match(reg, word):
			result.append(word)
	return result

def encode_word_with_dict(document, df_dict):
	result = []
	# print(df_dict)
	for word in document:
		if word in list(df_dict['word']):
			# If the word is in the dict
			word_id = df_dict.loc[(df_dict['word'] == word), 'id'].iloc[0]
		else:
			# If the word is not in the dict
			word_id = 0
		result.append(word_id)
	return result

def find_average_num_word(tokens_list):
	word_count_list = []
	for doc in tokens_list:
		word_count_list.append(len(doc))
	avg_num_word_in_doc = int(sum(word_count_list)/len(word_count_list))
	print("Average number of words in one document = ", avg_num_word_in_doc)
	return avg_num_word_in_doc

def strip_token_to_length(tokens_list, length):
	for doc in tokens_list:
		if len(doc) < length:
			N = length - len(doc)
			for i in range(N):
				doc.append(0)
		else:
			doc = doc[:length]
		# print(str(len(doc)))
	return tokens_list

def strip_token_to_length_df(tokens_df, length):
	return tokens_df.loc[:,:length].fillna(0)

def stemming_text(tokens_list):
	from nltk.stem.snowball import SnowballStemmer
	stemmer = SnowballStemmer("english")
	result = []
	for token in tokens_list:
		temp = stemmer.stem(token)
		result.append(temp)
	return result

def lemma_text(tokens_list):
	from nltk.stem import WordNetLemmatizer
	lemmatizer = WordNetLemmatizer()
	result = []
	for token in tokens_list:
		temp = lemmatizer.lemmatize(token)
		result.append(temp)
	return result

def n_gram(n, tokens_list):
	from nltk.util import ngrams
	ngrams = list(ngrams(tokens_list, n))
	return ngrams







