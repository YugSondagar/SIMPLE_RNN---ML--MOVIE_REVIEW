#Import libaries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


#Load the IMDB dataset
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

model = load_model('SimpleRNN_imdb.h5')

##Helper function
def decode(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

## Prediction Function
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.90 else 'Negative'
    return sentiment,prediction[0][0]

## Streamlit app

import streamlit as st
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="wide")

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as Positive or Negative.')



# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    sentiment,prediction = predict_sentiment(user_input)

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction}')
else:
    st.write("Please Enter a Movie Review.")
