import random
import numpy as np

VOCAB = {}
MAX_VOCAB_SIZE = 100000


def vectorize(example):
    """
    Place-holder implementation of vectorize; provides a unigram representation
    of the tweet example's text.
    """
    tokens = [token.lower() for token in example['text'].split()]
    for token in tokens:
        if token not in VOCAB:
            VOCAB[token] = len(VOCAB)
    vec = np.zeros(MAX_VOCAB_SIZE, dtype='float32')
    for token in tokens:
        vec[VOCAB[token]] += 1
    return vec


def train(labelled_vectors):
    """
    The train function will take in a list of the form yielded by
    `data.iter_vecs`, and will yield an object that implements the `predict`
    function, whose interface is shown by `Model`.
    """
    return Model(labelled_vectors)


class Model(object):

    def __init__(self, labelled_vectors):
        pass

    def predict(self, labelled_vector):
        """
        Takes in a triple (vectorized_tweet, target, None), and yields a 
        an pair of the form
            (stance_prediction, confidence)

        where stance prediction is 1:FAVOR, 0:AGAINST, -1:NEITHER
        """
        return [1, 0, -1][random.randint(0,2)], random.random()




