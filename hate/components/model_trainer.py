import os
import sys
import pickle
import pandas as pd

from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from hate.ml.model import ModelArchitecture
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import (
    ModelTrainerArtifacts,
    DataTransformationArtifacts,
)


class ModelTrainer:

    def __init__(
        self,
        data_transformation_artifacts: DataTransformationArtifacts,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def spliting_data(self, csv_path):

        try:
            logging.info("Reading transformed dataset")

            df = pd.read_csv(csv_path)

            # Keep only required columns
            df = df[[TWEET, LABEL]]

            # Remove missing values
            df = df.dropna(subset=[TWEET, LABEL])

            # Correct datatype
            df[TWEET] = df[TWEET].astype(str)
            df[LABEL] = df[LABEL].astype(int)

            x = df[TWEET]
            y = df[LABEL]

            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.30,
                random_state=self.model_trainer_config.RANDOM_STATE,
                stratify=y
            )

            return x_train, x_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys) from e

    def tokenizing(self, x_train):

        try:

            logging.info("Tokenizing data")

            x_train = x_train.fillna("").astype(str)

            tokenizer = Tokenizer(
                num_words=self.model_trainer_config.MAX_WORDS
            )

            tokenizer.fit_on_texts(x_train)

            sequences = tokenizer.texts_to_sequences(x_train)

            sequences_matrix = pad_sequences(
                sequences,
                maxlen=self.model_trainer_config.MAX_LEN
            )

            return sequences_matrix, tokenizer

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self):

        try:

            logging.info("Starting Model Trainer")

            x_train, x_test, y_train, y_test = self.spliting_data(
                self.data_transformation_artifacts.transformed_data_path
            )

            model = ModelArchitecture().get_model()

            sequences_matrix, tokenizer = self.tokenizing(x_train)

            model.fit(
                sequences_matrix,
                y_train,
                batch_size=self.model_trainer_config.BATCH_SIZE,
                epochs=self.model_trainer_config.EPOCH,
                validation_split=self.model_trainer_config.VALIDATION_SPLIT,
                verbose=1
            )

            os.makedirs(
                self.model_trainer_config.TRAINED_MODEL_DIR,
                exist_ok=True,
            )

            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)

            with open("tokenizer.pickle", "wb") as file:
                pickle.dump(tokenizer, file)

            # -----------------------------
            # Convert Series -> DataFrame
            # -----------------------------
            x_train = pd.DataFrame({
                TWEET: x_train.values
            })

            x_test = pd.DataFrame({
                TWEET: x_test.values
            })

            y_test = pd.DataFrame({
                LABEL: y_test.values
            })

            # -----------------------------
            # Save CSV
            # -----------------------------
            x_train.to_csv(
                self.model_trainer_config.X_TRAIN_DATA_PATH,
                index=False
            )

            x_test.to_csv(
                self.model_trainer_config.X_TEST_DATA_PATH,
                index=False
            )

            y_test.to_csv(
                self.model_trainer_config.Y_TEST_DATA_PATH,
                index=False
            )

            logging.info("Model training completed successfully.")

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH,
            )

            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e