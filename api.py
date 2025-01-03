from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# nltk.download('stopwords')

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])

def predict():
    cv = pickle.load(open(r"./count_vectorizer.pkl", "rb"))
    predictor = pickle.load(open(r"./gb_model.pkl", "rb"))
    try:
        if "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment = make_prediction(predictor, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})

def make_prediction(predictor, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    y_predictions = predictor.predict_proba(X_prediction)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

if __name__ == "__main__":
    app.run(port=8080, debug=True)