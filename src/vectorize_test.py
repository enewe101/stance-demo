import numpy as np
import os
import twokenize
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from unigram_dictionary import UnigramDictionary
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

#TODO: build the dictionaries and train the word2vec model

class Vectorizer(object):
	def __init__(
		self,
		char_ngram=None,
		word_ngram=None,
		word_vectors=None
	):
		self.char_ngram = char_ngram
		self.word_ngram = word_ngram
		self.word_vectors = word_vectors

	def vectorize(self, text):
		"""
		Returns the feature vector for a given text
		"""
		word_tokens = twokenize.tokenize(text)
		char_tokens = list(text)

		#Don't do anything if the necessary data isnt there
		if self.char_ngram or self.word_ngram or self.word_vectors is None:
			print("Missing dictionary or word embeddings")
			return

		word_features = find_ngram_ft_vec(word_tokens, self.word_ngram)
		char_features = find_ngram_ft_vec(char_tokens, self.char_ngram)

		# total_vector = None
		# count = 0
		# for w in word_tokens:
		# 	if w in self.word_vectors:
		# 		count += 1
		# 		if total_vector is None:
		# 			total_vector = self.word_vectors[w]
		# 		else:
		# 			total_vector += self.word_vectors[w]

		# word_embding = coo_matrix(np.divide(total_vector, count))

		feature_vect = hstack((word_features, char_features))
		#feature_vect = hstack((feature_vect, word_embding))

		return feature_vect

	def build_dict(self, corpus, word=True, which_grams=None):
		"""
		Builds the necessary ngrams out of the corpus
		Word is set to True by default which builds word ngrams
		If set to False, will build character ngrams
		Which_grams is the n values of the ngrams. By default will
		create unigrams, bigrams, and trigrams for words and
		bigrams, trigrams, four-grams and five-grams for characters
		"""
		dct = UnigramDictionary()
		if word:
			which_grams = [1,2,3]
		else:
			which_grams = [2,3,4,5]

		for text in corpus:
			#list of tokens, each index is the zipped object of tokens for the given n
			all_tokens = [ find_ngams(tokens, n) for n in which_grams ]

			for n in all_tokens:
				for token in n:
					dct.add(token)

		if word:
			self.word_ngram = dct
		else:
			self.char_ngram = dct

		return dct

	def train_word2vec(self, domain_corpus, feature_length, sg=1, min_count=5, workers=3):
		"""
		Trains the word2vec model on a corpus, by default using the skip-gram model
		"""
		tokenized = [ twokenize.tokenize(text) for text in domain_corpus ]
		model = Word2Vec(tokenized, min_count=min_count, sg=1, workers=workers)
		word_vectors = model.wv
		del model
		self.word_vectors = word_vectors

	def save_word2vec(self, fname):
		"""
		Saves the word vectors
		"""
		if not os.path.isdir(dir_name):
			os.mkdir(dir_name)

		self.word_vectors.save(fname)

	def load_word2vec(self, fname):
		"""
		Loads the word vectors
		"""
		self.word_vectors = KeyedVectors.load(fname)

	def save_dicts(self, dir_name):
		"""
		Saves both ngrams into a directory called fname
		"""
		if not os.path.isdir(dir_name):
			os.mkdir(dir_name)

		self.char_ngram.save("{0}/{1}".format(fname, "char_ngram"))
		self.word_ngram.save("{0}/{1}".format(fname, "word_ngram"))

	def load_dicts(self, dir_name):
		"""
		Loads both ngrams, overwrites current data
		"""
		self.char_ngram = UnigramDictionary()
		self.char_ngram.load("{0}/{1}".format(fname, "char_ngram"))

		self.word_ngram = UnigramDictionary()
		self.word_ngram.load("{0}/{1}".format(fname, "word_ngram"))

def find_ngram_ft_vec(tokens, ngram_dict, which_grams=[1,2,3]):
	"""
	Give a list of tokens, creates the feature vector for given ngrams
	Which is the absence or presence of contiguous ngrams
	"""
	occ_vector = coo_matrix((1, len(ngram_dict)), dtype=int)
	occ_vector = occ_vector.tocsr()

	#list of tokens, each index is the zipped object of tokens for the given n
	all_tokens = [ find_ngrams(tokens, n) for n in which_grams ]

	for n in all_tokens:
		for token in n:
			occ_vector[0, ngram_dict.get_id(token)] += 1

	return occ_vector

def find_ngrams(tokens, n):
	"""
	Returns a zip object of the ngram tokens for a give list of tokens
	"""
	return zip(*[input_list[i:] for i in range(n)])



