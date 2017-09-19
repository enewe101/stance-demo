from vectorize_test import Vectorizer
from data import corpus
from SETTINGS import NGRAMS_DIR

#TODO: build the dictionaries and train the word2vec model


def prepare_vectorizer():
    vectorizer = Vectorizer()
    vectorizer.build_dict(corpus(), word=True)
    vectorizer.build_dict(corpus(), word=False)
    vectorizer.save_dicts(NGRAMS_DIR)


if __name__ == '__main__':
    prepare_vectorizer()


