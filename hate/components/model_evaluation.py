import os
import sys
import pickle
import keras
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix

from hate.logger import logging
from hate.exception import CustomException
from hate.constants import *
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import (
    ModelEvaluationArtifacts,
    ModelTrainerArtifacts,
    DataTransformationArtifacts,
)


class ModelEvaluation:

    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifacts: ModelTrainerArtifacts,
        data_transformation_artifacts: DataTransformationArtifacts,
    ):

        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts

    def evaluate(self):

        try:

            logging.info("Loading test data")

            x_test = pd.read_csv(
                self.model_trainer_artifacts.x_test_path
            )

            y_test = pd.read_csv(
                self.model_trainer_artifacts.y_test_path
            )

            print(x_test.head())
            print(y_test.head())

            # Get tweet column safely
            if TWEET in x_test.columns:
                x_test = x_test[TWEET]
            else:
                x_test = x_test.iloc[:, 0]

            # Get label column safely
            if LABEL in y_test.columns:
                y_test = y_test[LABEL]
            else:
                y_test = y_test.iloc[:, 0]

            x_test = x_test.fillna("").astype(str)
            y_test = y_test.astype(int)

            with open("tokenizer.pickle", "rb") as f:
                tokenizer = pickle.load(f)

            model = keras.models.load_model(
                self.model_trainer_artifacts.trained_model_path
            )

            sequences = tokenizer.texts_to_sequences(x_test)

            sequences = pad_sequences(
                sequences,
                maxlen=MAX_LEN
            )

            loss, accuracy = model.evaluate(
                sequences,
                y_test,
                verbose=0
            )

            print(f"\nAccuracy : {accuracy:.4f}")

            prediction = model.predict(sequences)

            prediction = (prediction > 0.5).astype(int)

            print(
                confusion_matrix(
                    y_test,
                    prediction
                )
            )

            return accuracy

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self):

        try:

            logging.info("Starting model evaluation")

            accuracy = self.evaluate()

            print(f"\nModel Accuracy : {accuracy}")

            # Since we are not using Google Cloud,
            # always accept the newly trained model.
            model_evaluation_artifact = ModelEvaluationArtifacts(
                is_model_accepted=True
            )

            return model_evaluation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e