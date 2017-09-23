# Training file
from vectorize_test import Vectorizer
from data import corpus
from data import iter_vecs
from sklearn import svm
from SETTINGS import NGRAMS_DIR
import pickle as pkl
from scipy.sparse import coo_matrix, hstack, vstack, csr_matrix
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import classification_report
import logging
.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DUMB_MODEL = 'dumb_model.pkl'


def preprocess(vecs, train=True):
    """ add the features into a single sparse matrix for training
        if Train=False, then dont bother about y
    """
    X_1 = [p[0] for p in vecs]
    X_2 = np.array([p[1] for p in vecs])
    logging.debug(X_1[0].shape)
    X_1 = vstack(X_1)
    logging.debug(X_1.shape)
    logging.debug(X_2.shape)
    X = hstack((X_1, csr_matrix(X_2)))
    logging.debug(X.shape)
    if train:
        y = [p[2] for p in vecs]
        return X, y
    else:
        return X
 

def train_svm(load_vecs=True):
    """ Train the model using SVM
    """
    clf = svm.SVC(kernel='linear')
    if not load_vecs:
        train_vecs = iter_vecs()
        pkl.dump(train_vecs, open('../data/train_vecs_2.pkl', 'wb'))
    else:
        train_vecs = pkl.load(open('../data/train_vecs_2.pkl', 'rb')) 
    logging.info('Got training vecs')
    if not load_vecs:
        test_vecs = iter_vecs(include_train=False, include_test=True)
        pkl.dump(test_vecs,open('../data/test_vecs_2.pkl','wb'))
    else:
        test_vecs = pkl.load(open('../data/test_vecs_2.pkl','rb'))
    logging.info('Got testing vecs')
    X_train,y_train = preprocess(train_vecs)
    X_test,y_test = preprocess(test_vecs)
    logging.info(clf)
    logging.info('Starting training')
    clf.fit(X_train,y_train)
    joblib.dump(clf,DUMB_MODEL)

    yp = clf.predict(X_test)
    metrics(y_test, yp)


def predict(vecs):
    """ Given the output of vectorizer, predict the y's
    """
    X = preprocess(vecs,train=False)
    clf = joblib.load(DUMB_MODEL)
    yp = clf.predict(X)
    yprob = clf.predict_proba(X)
    return yp, yprob


def metrics(y_true, y_pred):
    """ Implement metrics here
    """
    acc = np.mean(y_true == y_pred)
    logging.info(acc)
    logging.info(classification_report(y_true, y_pred))


if __name__ == '__main__':
    train_svm()


