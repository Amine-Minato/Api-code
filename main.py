import re
import pickle
from string import punctuation
import uvicorn
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow_hub as hub
from pyngrok import ngrok
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI

app = FastAPI()

def download_nltk_resources():
    nltk.download("stopwords")
    nltk.download("wordnet")

def initialize_nltk_resources():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    return stop_words, lemmatizer

stop_words, lemmatizer = initialize_nltk_resources()

def feature_USE_fct(sentences):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    features_list = []
    feat = embed(sentences)
    return feat

def text_cleaning(text, remove_stop_words=True, lemmatize_words=False):
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"'s", " ", text)
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

loaded_model = None
mlb = None

@app.on_event("startup")
async def startup_event():
    download_nltk_resources()
    global loaded_model, mlb
    with open('saved_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)

@app.get("/predict-tags")
def predict_tags(tags: str):
    cleaned_tags = text_cleaning(tags)
    
    text_vector_use = feature_USE_fct([cleaned_tags])

    prediction = loaded_model.predict(text_vector_use)

    tags = mlb.inverse_transform(prediction)
    return {'predict-tags': tags}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
