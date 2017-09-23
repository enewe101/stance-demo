import os
import csv
import requests
import itertools
import numpy as np
from interface_examples import vectorize
from SETTINGS import DATA_DIR, TWEET_ENDPOINT

UNIVERSAL_NEWLINE_MODE = 'rU'
VOCAB = {}
MAX_VOCAB_SIZE = 100000

targets = {
        'Atheism':0,
        'Climate Change is a Real Concern': 1,
        'Feminist Movement': 2,
        'Hillary Clinton' : 3,
        'Legalization of Abortion': 4
}

def iter_raw(include_train=True, include_test=False):
    """
    Yield the dataset (testing and/or training) in the form of an iterable of
    tweet dicts, each having keys 'text', 'stance' (i.e. stance), and 'target'.
    """

    train_data, test_data = [], []

    if include_train:
        print('starting to read')
        train_path = os.path.join(DATA_DIR, 'train.csv')
        train_data = csv.DictReader(open(
            train_path, UNIVERSAL_NEWLINE_MODE, encoding="ISO-8859-1"))
        print('reading complete')

    if include_test:
        print("starting to read test")
        test_path = os.path.join(DATA_DIR, 'test.csv')
        test_data = csv.DictReader(open(
            test_path, UNIVERSAL_NEWLINE_MODE, encoding="ISO-8859-1"))
        print("reading complete")

    return [
        {
            'text': example['Tweet'],
            'stance': convert_stance(example['Stance']),
            'target': example['Target']
        }
        for example in itertools.chain(train_data, test_data)
    ]


def populate_database(endpoint=TWEET_ENDPOINT):
    for tweet in iter_raw():
        response = requests.post(TWEET_ENDPOINT, tweet)
        response.raise_for_status()
        print(response.text)


def convert_stance(stance):
    if stance == 'AGAINST':
        return -1
    elif stance == 'NONE':
        return 0
    elif stance == 'FAVOR':
        return 1
    else:
        raise ValueError('Unexpected value for stance.')

def convert_target(target):
    one_hot = np.zeros(len(targets))
    one_hot[targets[target]] = 1
    return one_hot

def corpus(include_train=True, include_test=False):
    for tweet in iter_raw(include_train, include_test):
        yield tweet['text']


def iter_vecs(include_train=True, include_test=False):
    """
    Yield an iterable of vectorized examples.  Each example is a tuple, with
    the form:
                (feature_vector, target, stance)
    """

    vecs = []
    ct = 0
    for example in iter_raw(include_train, include_test):
        vecs.append(
            (vectorize(example), convert_target(example['target']), example['stance'])
        )
        ct += 1
        print('done %d' % ct)
    return vecs


