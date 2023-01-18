import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

# loading in the model to predict on the data
pickle_in = open('languageModel_MNB.pkl', 'rb')
classifier = pickle.load(pickle_in)

pickle_cv = open('transform_languageModel_MNB.pkl', 'rb')
CV = pickle.load(pickle_cv)

pickle_encoder = open('encoder_languageModel_MNB.pkl', 'rb')
encoder = pickle.load(pickle_encoder)

def welcome():
	return 'welcome all'


# defining the function which will make the prediction using
# the data which the user inputs

def prediction(text):
    x= CV.transform([text]).toarray()
    lang= classifier.predict(x)
    lang= encoder.inverse_transform(lang)
    print("This word/sentence contains {} word(s).".format(lang[0]))
    return (lang[0])


	

# this is the main function in which we define our webpage
def main():
    st.title('Language Detection tool')

    input_text = st.text_input("Text", "Type here")
    result = ""

    if st.button("Predict"):
        result = prediction(input_text)
    st.success('The output is {}'.format(result))

if __name__ == '__main__':
    main()
