import os
import re
import sys
import string
import pandas as pd
import nltk
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
)


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifacts: DataIngestionArtifacts,
    ):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def raw_data_cleaning(self):
        try:
            logging.info("Reading dataset")

            df = pd.read_csv(
                self.data_ingestion_artifacts.raw_data_file_path
            )

            # Drop unwanted columns if they exist
            cols_to_drop = [
                col
                for col in self.data_transformation_config.DROP_COLUMNS
                if col in df.columns
            ]

            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)

            # Convert labels
            df[self.data_transformation_config.CLASS].replace(
                {0: 1, 2: 0},
                inplace=True,
            )

            # Rename class -> label
            df.rename(
                columns={
                    self.data_transformation_config.CLASS:
                    self.data_transformation_config.LABEL
                },
                inplace=True,
            )

            return df

        except Exception as e:
            raise CustomException(e, sys) from e

    def concat_data_cleaning(self, text):
        try:
            stemmer = nltk.SnowballStemmer("english")
            stop_words = set(stopwords.words("english"))

            text = str(text).lower()

            text = re.sub(r"\[.*?\]", "", text)
            text = re.sub(r"https?://\S+|www\.\S+", "", text)
            text = re.sub(r"<.*?>+", "", text)
            text = re.sub(
                "[%s]" % re.escape(string.punctuation),
                "",
                text,
            )
            text = re.sub(r"\n", "", text)
            text = re.sub(r"\w*\d\w*", "", text)

            words = [
                word
                for word in text.split()
                if word not in stop_words
            ]

            words = [stemmer.stem(word) for word in words]

            return " ".join(words)

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifacts:

        try:
            logging.info("Starting Data Transformation")

            df = self.raw_data_cleaning()

            # Remove missing tweets
            df = df.dropna(
                subset=[self.data_transformation_config.TWEET]
            )

            # Convert tweets to string
            df[self.data_transformation_config.TWEET] = (
                df[self.data_transformation_config.TWEET]
                .astype(str)
                .apply(self.concat_data_cleaning)
            )

            os.makedirs(
                self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                exist_ok=True,
            )

            df.to_csv(
                self.data_transformation_config.TRANSFORMED_FILE_PATH,
                index=False,
            )

            logging.info("Transformation completed")

            return DataTransformationArtifacts(
                transformed_data_path=self.data_transformation_config.TRANSFORMED_FILE_PATH
            )

        except Exception as e:
            raise CustomException(e, sys) from e