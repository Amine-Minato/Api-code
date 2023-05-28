import re
import tensorflow as tf
import pickle
from string import punctuation
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import tensorflow_hub as hub
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

def initialize_nltk_resources():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

stop_words, lemmatizer = initialize_nltk_resources()

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

app = FastAPI(
    title="Suggestion Tags API",
    description="A simple API that uses an NLP model to predict tag suggestions",
    version="0.1",
)

#app.add_middleware(
    #CORSMiddleware,
    #allow_origins=["*"],
    #allow_credentials=True,
    #allow_methods=["*"],
    #allow_headers=["*"],
#)


with open('saved_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('mlb.pkl', 'rb') as f:
    mlb = pickle.load(f)


def feature_USE_fct(sentences, b_size=1):
    batch_size = b_size
    features_list = []
    feat = embed(sentences)
    return feat


def text_cleaning(text, remove_stop_words=True, lemmatize_words=False):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    
    text = "".join([c for c in text if c not in punctuation])
    
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    if lemmatize_words:
        text = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    return text


@app.get("/predict-tags")
def predict_tags(tags: str):
    cleaned_tags = text_cleaning(tags)
    text_vector_use = feature_USE_fct([cleaned_tags])
    
    prediction = loaded_model.predict(text_vector_use)

    tags = mlb.inverse_transform(prediction)
    return {'predict-tags': tags}
