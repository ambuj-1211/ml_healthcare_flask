from preprocessor import preprocessor
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import one_hot
import pickle



def predict(text):
    #now it will predict the disease
    preproc=preprocessor()
    text=preproc.forward(text)
    model=pickle.load(open('model.pkl','rb'))
    return list(model.predict(text))
