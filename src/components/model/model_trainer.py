import numpy as np
import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error,root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.components.handling.exceptions import CustomException
from src.components.handling.logger import logging
from src.components.handling.utils import save_obj, evaluate_model



@dataclass
class ModelConfig:
    trained_model_path = os.path.join('outputs','trained_model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_config = ModelConfig()

    
    def start_trainer(self,train_set,test_set):
        try:
            logging.info("Initializing train and test sets")
            X_train,y_train,X_test,y_test = (
                train_set[:,:-1],
                train_set[:,-1],
                test_set[:,:-1],
                test_set[:,-1]
            )

            models = {
                "Random Forests" : RandomForestRegressor(),
                "AdaBoost" : AdaBoostRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Linear Regressor" : LinearRegression(),
                "XGB Regressor" : XGBRegressor(),
                "CatBoost" : CatBoostRegressor(), 
                "KNeighbours Regressor" : KNeighborsRegressor()
            }

            logging.info("Evaluating all the models...")

            model_report:dict = evaluate_model(X_train= X_train,y_train = y_train,X_test = X_test ,y_test = y_test, models = models,eval_metric=r2_score)

            logging.info("Model evaluation complete. Determining best model...")
            best_score = max(sorted(model_report.values()))

            if best_score < 0.6 :
                raise CustomException("No model is good for this data.")
            
            logging.info("Best model found.")


            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]

            logging.info("Best model is {} with score {}".format(best_model_name, best_score))
            best_model = models[best_model_name]
            save_obj(
                filepath= self.model_config.trained_model_path,
                obj=best_model

            )



        except Exception as e:
            raise CustomException(e,sys)




