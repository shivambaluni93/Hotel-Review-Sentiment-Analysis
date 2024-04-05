#!/usr/bin/env python
# coding: utf-8
# In[ ]:
#!/usr/bin/env python
# coding: utf-8
# In[1]:

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
import numpy as np
import time
from PIL import Image
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
import base64

im = Image.open('icon.png')
st.set_page_config(page_title="Review Classifier",page_icon=im)

html_temp = """
    <div style="background-color:#f63350 ;padding:10px">
    <h1 style="color:white;text-align:center;">
    Hotel Review Classification App </h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
st.image('main.jpg')

model = load(open('classify.sav', 'rb'))
tokenizer = Tokenizer(num_words=10000)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = load(f)

def predict_sentiment(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequences = pad_sequences(sequences, maxlen=100)
    prediction = model.predict(padded_sequences)
    predicted_label = np.argmax(prediction, axis=1)[0]
    sentiment_classes = {0: 'Negative', 1: 'Positive'}
    predicted_sentiment = sentiment_classes[predicted_label]
    return predicted_sentiment

def main():
    st.header("How was your experience at our hotel?")
    review = st.text_area("", "Enter your review here....", height=180) 
    if st.button("Predict"):
        if review.strip() == "":
            st.warning("Please enter a review.")
        else:
            output = predict_sentiment(review)
            if output=='Positive':
                st.success(f"The sentiment of the review is: {output}😄")
                st.markdown("![Alt Text](https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif)")
            else:
                st.success(f"The sentiment of the review is: {output}😒")
                st.markdown("![Alt Text](https://media.giphy.com/media/eaSOfHokcygg8ogGOF/giphy.gif)")

if __name__ == '__main__':
    main()
    
# In[ ]:





