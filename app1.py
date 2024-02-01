# app.py
import pandas as pd
import streamlit as st
from joblib import load
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

st.title(":orange[NLP]")
st.header(":green[Financial Sentiment Analysis]")

user_input = st.text_input('Enter Sentence:', max_chars=500)

data = {'cleaned_sentence': user_input}
df = pd.DataFrame(data, index=[0])

df['cleaned_sentence'] = df['cleaned_sentence'].str.replace('-', ' ')

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

def extract_nouns(tokens):
    tagged_tokens = pos_tag(tokens)
    nouns = [token[0] for token in tagged_tokens if token[1].startswith('N')]
    return nouns

def preprocess_text(text):
    # Check if the input is not empty and is a string
    if isinstance(text, str):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word.lower() not in stop_words]
        nouns = extract_nouns(tokens)
        return ' '.join(nouns)
    else:
        return ''


if st.button('Predict'):
    loaded_model = load(r"C:\Users\kkoll\PycharmProjects\NLP Financial Sentimental analysis\xgb_model.sav")

    prediction = loaded_model.predict(df['cleaned_sentence'])

    st.subheader('Predicted Result')

    if prediction == 2:
        st.success("Positive sentiment.")
    if prediction == 0:
        st.error("Negative sentiment.")
    if prediction == 1:
        st.info("Neutral sentiment.")
