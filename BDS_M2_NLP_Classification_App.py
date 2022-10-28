import streamlit as st
import pandas as pd
import numpy as np
import pickle
#import preprocessor as prepro

import spacy
nlp = spacy.load('en_core_web_sm')

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import eli5

st.set_page_config(
    page_title="Pol-Finder 🔍",
    page_icon="🔍")

# Defining a function to preprocess data
def text_prepro(texts):
  """
  takes in a pandas series (1 column of a DF)
  removes twitter stuff
  lowercases, normalizes text
  """
  texts_clean = texts.map(lambda t: prepro.clean(t))
  texts_clean = texts_clean.str.replace('#','')

  clean_container = []

  for text in nlp.pipe(texts_clean, disable=["tagger", "parser", "ner"]):

    txt = [token.lemma_.lower() for token in text 
          if token.is_alpha 
          and not token.is_stop 
          and not token.is_punct]

    clean_container.append(" ".join(txt))
    pbar.update(1)
  
  return clean_container

@st.experimental_singleton
def load_model():
    pipe = pickle.load(open('pol_model_pipe.pkl','rb'))
    return pipe

pipe = load_model()

txt = st.text_area('Text to analyze', '''
Write here some political text
    ''')

if st.button('Predict pol party 😵'):
    to_analyse = text_prepro(pd.Series([txt]))
    result = pipe.predict(to_analyse)[0]
    st.write(['Republican!','Democrat'][result])