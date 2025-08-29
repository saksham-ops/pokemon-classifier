import os,sys
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from imblearn.combine import SMOTETomek
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    Preprocessor_path = os.path.join("artifacts", "Preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_preprocessor_obj(self):
        
        logging.info("Data Preprocessing Starts")

        try:

            preprocessor = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            logging.info("Preprocessor object obtained")

            return preprocessor
        
        except Exception as e:
            logging.info('Exception occured during Data Preprocessing')
            raise CustomException(e,sys)


    def initiate_DataTransformation(self,train_data,test_data):
        
        try:
            train_data = pd.read_csv(train_data)
            test_data = pd.read_csv(test_data)

            x_train = train_data.drop("isLegendary", axis=1)
            y_train = train_data["isLegendary"]
            x_test = test_data.drop("isLegendary", axis=1)
            y_test = test_data["isLegendary"]

            y_train = y_train.map(lambda x: 1 if x else 0)
            y_test = y_test.map(lambda x: 1 if x else 0)
            x_train["hasMegaEvolution"] = x_train["hasMegaEvolution"].map(lambda x: 1 if x else 0)
            x_test["hasMegaEvolution"] = x_test["hasMegaEvolution"].map(lambda x: 1 if x else 0)

            for col in x_train.columns:
                if col=="Unnamed: 0":
                    x_train.drop(col,axis=1,inplace=True)

            for col in x_test.columns:
                if col=="Unnamed: 0":
                    x_test.drop(col,axis=1,inplace=True)

            preprocessor_obj = self.get_preprocessor_obj()
            x_train_transformed = preprocessor_obj.fit_transform(x_train)
            x_test_transformed = preprocessor_obj.transform(x_test)

            smt= SMOTETomek(sampling_strategy="minority")
            train_x, train_y = smt.fit_resample(x_train_transformed,y_train)
            test_x, test_y = smt.fit_resample(x_test_transformed,y_test)


            save_object(self.transformation_config.Preprocessor_path, preprocessor_obj)

            logging.info("Data transformaion Completed")

            return train_x,test_x,train_y,test_y,self.transformation_config.Preprocessor_path

        except Exception as e:
            logging.info('Exception occured during Data Transformation')
            raise CustomException(e,sys)






