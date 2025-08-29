from dataclasses import dataclass
import os,sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    Raw_data_path = os.path.join("artifacts", "Raw_data.csv")
    Train_data_path = os.path.join("artifacts", "Train_data.csv")
    Test_data_path = os.path.join("artifacts", "Test_data.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_DataIngestion(self):

        logging.info("Data Ingestion Started")

        try:
            df = pd.read_csv("notebook\pokemon_alopez247.csv")

            os.makedirs(os.path.dirname(self.ingestion_config.Raw_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.Raw_data_path)

            df = df.drop(labels=df.columns[df.isna().sum() > 0.25*len(df)], axis=1)
            df.drop(labels=["Name","Number","Type_1","Egg_Group_1","Body_Style","Color","Generation","hasGender","Pr_Male"],axis=1,inplace=True)

            x_train, x_test = train_test_split(df,test_size=0.25,random_state=42)

            x_train.to_csv(self.ingestion_config.Train_data_path)
            x_test.to_csv(self.ingestion_config.Test_data_path)

            logging.info("Data Ingestion Completed")

            return self.ingestion_config.Raw_data_path, self.ingestion_config.Train_data_path, self.ingestion_config.Test_data_path
        
        except Exception as e:
            logging.info('Exception occured during Data Ingestion')
            raise CustomException(e,sys)
    