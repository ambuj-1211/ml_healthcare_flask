import pickle

from preprocessor import preprocessor


def predict(text):
    #now it will predict the disease
    preproc=preprocessor()
    text=preproc.forward(text)
    model=pickle.load(open('model.pkl','rb'))
    return list(model.predict(text))
