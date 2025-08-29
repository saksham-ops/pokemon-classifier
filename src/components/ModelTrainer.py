import os, sys
from src.logger import logging
from dataclasses import dataclass
from src.exception import CustomException
from src.utils import save_object, load_object, evaluate_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score


@dataclass
class ModelTrainerConfig:
    model_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.Modelconfig = ModelTrainerConfig()

    def Initiate_model_training(self,x_train,x_test,y_train,y_test):

        logging.info("model training starts")
        
        try:
            models={"RandomForest": RandomForestClassifier(),"DecisionTree": DecisionTreeClassifier(), "AdaBoost": AdaBoostClassifier(algorithm="SAMME"), "GradientBoost": GradientBoostingClassifier()}
            
            model_score = evaluate_model(x_train, y_train, x_test,y_test, models)
            
            model_name = list(model_score.keys())[list(model_score.values()).index(max(list(model_score.values())))]
            best_model = models[model_name]

            save_object(self.Modelconfig.model_path, best_model)

            logging.info("model saved")

            return self.Modelconfig.model_path

        except Exception as e:
            logging.info('Exception occured during model training')
            raise CustomException(e,sys)

