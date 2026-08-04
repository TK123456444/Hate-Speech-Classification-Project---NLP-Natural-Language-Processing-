import os
import sys
import re
import string
import pickle

import nltk
from nltk.corpus import stopwords

import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

from hate.logger import logging
from hate.exception import CustomException
from hate.constants import MAX_LEN

nltk.download("stopwords", quiet=True)


class PredictionPipeline:

    def __init__(self):

        BASE_DIR = os.getcwd()

        # ----------------------------
        # FIND MODEL AUTOMATICALLY
        # ----------------------------

        self.model_path = None

        for root, dirs, files in os.walk(
            os.path.join(BASE_DIR, "artifacts")
        ):
            for file in files:

                if file.endswith(".h5") or file.endswith(".keras"):

                    self.model_path = os.path.join(root, file)
                    break

            if self.model_path:
                break

        if self.model_path is None:
            raise Exception(
                "No trained model (.h5/.keras) found inside artifacts folder."
            )

        # ----------------------------
        # TOKENIZER
        # ----------------------------

        self.tokenizer_path = os.path.join(
            BASE_DIR,
            "tokenizer.pickle"
        )

        if not os.path.exists(self.tokenizer_path):
            raise Exception("tokenizer.pickle not found.")

    # ---------------------------------------------------------

    def clean_text(self, text):

        stemmer = nltk.SnowballStemmer("english")
        stop_words = set(stopwords.words("english"))

        text = str(text).lower()

        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>+", "", text)
        text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
        text = re.sub(r"\n", "", text)
        text = re.sub(r"\w*\d\w*", "", text)

        words = []

        for word in text.split():

            if word not in stop_words:

                words.append(stemmer.stem(word))

        return " ".join(words)

    # ---------------------------------------------------------

    def predict(self, text):

        try:

            logging.info(f"Loading model from : {self.model_path}")

            model = keras.models.load_model(self.model_path)

            with open(self.tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)

            text = self.clean_text(text)

            sequence = tokenizer.texts_to_sequences([text])

            padded = pad_sequences(
                sequence,
                maxlen=MAX_LEN
            )

            prediction = model.predict(padded, verbose=0)

            score = float(prediction[0][0])

            print("Prediction Score :", score)

            if score >= 0.5:
                return {
                    "prediction": "hate and abusive",
                    "score": score
                }

            return {
                "prediction": "no hate",
                "score": score
            }

        except Exception as e:
            raise CustomException(e, sys)

    # ---------------------------------------------------------

    def run_pipeline(self, text):

        return self.predict(text)