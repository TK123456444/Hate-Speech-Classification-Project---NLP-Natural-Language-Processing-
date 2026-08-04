import os
import sys
import shutil

from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Data Ingestion using local dataset.
        """
        self.data_ingestion_config = data_ingestion_config

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Copies the local dataset into the artifacts folder.
        """

        logging.info("Entered initiate_data_ingestion")

        try:
            # Create artifacts directory
            os.makedirs(
                self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,
                exist_ok=True,
            )

            # Local dataset path
            source_file = os.path.join(
                os.getcwd(),
                "data",
                "labeled_data.csv"
            )

            if not os.path.exists(source_file):
                raise FileNotFoundError(
                    f"Dataset not found at: {source_file}"
                )

            # Destination files
            imbalance_destination = self.data_ingestion_config.DATA_ARTIFACTS_DIR
            raw_destination = self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR

            # Copy dataset twice because the existing pipeline expects two files
            shutil.copy(source_file, imbalance_destination)
            shutil.copy(source_file, raw_destination)

            logging.info("Local dataset copied successfully.")

            return DataIngestionArtifacts(
                imbalance_data_file_path=imbalance_destination,
                raw_data_file_path=raw_destination,
            )

        except Exception as e:
            raise CustomException(e, sys) from e