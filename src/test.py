import data
import types
import numpy as np

# Michel, change this to import your assets
from interface_examples import vectorize

# Koustuv, change this to import your assets
from interface_examples import train


def test_vectorize():
    expected_type = np.array([]).__class__
    for example in data.iter_raw():
        vec = vectorize(example)
        try:
            assert isinstance(vec, expected_type), 'bad vectorize return type'
        except Exception:
            print "vectorize: bad return type"
            return
    print 'vectorize OK'


def test_train():
   model = train(data.iter_vecs()) 
   message = "train: should yield model that implements predict"
   try:
       assert isinstance(model.predict, types.MethodType), message
   except Exception:
       print "train: bad train return type"
       return

   print 'train OK'


def test_predict():
    model = train(data.iter_vecs()) 
    for test_example in data.iter_vecs(include_train=False, include_test=True):

        try:
            prediction, confidence =  model.predict(test_example)
            assert prediction in {-1,0,1}, "predict: invalid prediction"
        except Exception:
            print "predict: bad prediction return type"
            return

        try:
            assert confidence>0 and confidence<1, "predict: invalid confidence"
        except Exception:
            print "predict: bad prediction return type"
            return

    print 'predict OK'
   

if __name__ == '__main__':
    test_vectorize()
    test_train()
    test_predict()


