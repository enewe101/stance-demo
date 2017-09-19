import os
import csv
import itertools
import numpy as np
from interface_examples import vectorize


DATA_DIR = os.path.abspath(
    os.path.join(os.path.realpath(__file__), '../../data')
)
UNIVERSAL_NEWLINE_MODE = 'rU'
VOCAB = {}
MAX_VOCAB_SIZE = 100000


def iter_raw(include_train=True, include_test=False):
    """
    Yield the dataset (testing and/or training) in the form of an iterable of
    tweet dicts, each having keys 'text', 'label' (i.e. stance), and 'target'.
    """

    train_data, test_data = [], []

    if include_train:
        train_path = os.path.join(DATA_DIR, 'train.csv')
        train_data = csv.DictReader(open(
            train_path, UNIVERSAL_NEWLINE_MODE, encoding="ISO-8859-1"))

    if include_test:
        test_path = os.path.join(DATA_DIR, 'test.csv')
        test_data = csv.DictReader(open(
            test_path, UNIVERSAL_NEWLINE_MODE, encoding="ISO-8859-1"))

    return [
        {
            'text': example['Tweet'],
            'label': example['Stance'],
            'target': example['Target']
        }
        for example in itertools.chain(train_data, test_data)
    ]


def corpus(include_train=True, include_test=False):
    for tweet in iter_raw(include_train, include_test):
        yield tweet['text']


def iter_vecs(include_train=True, include_test=False):
    """
    Yield an iterable of vectorized examples.  Each example is a tuple, with
    the form:
                (feature_vector, target, stance)
    """

    return [
        (vectorize(example), example['target'], example['label'])
        for example in iter_raw(include_train, include_test)
    ]


