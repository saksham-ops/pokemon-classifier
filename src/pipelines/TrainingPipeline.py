import os, sys
from src.components.DataIngestion import DataIngestion
from src.components.DataTransformation import DataTransformation
from src.components.ModelTrainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class Training_pipeline:
    def __init__(self):
        pass

    def run_training_pipeline(self):

        try:
            data_ingestion = DataIngestion()
            raw_data, train_data, test_data = data_ingestion.initiate_DataIngestion()

            data_transfomation = DataTransformation()
            x_train, x_test, y_train, y_test, preprocessor_path = data_transfomation.initiate_DataTransformation(train_data,test_data)

            model_train = ModelTrainer()
            model_train.Initiate_model_training(x_train,x_test,y_train,y_test)

            logging.info("training Completed")

        except Exception as e:
            logging.info("Error occurred in training pipeline")
            raise CustomException(e,sys)

