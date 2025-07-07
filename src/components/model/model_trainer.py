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

            params ={
                "Random Forests" : {
                    'n_estimators' : [10, 50, 100, 250]
                },
                'AdaBoost' : {

                    'learning_rate' : [1,0.3,0.1,0.03,0.01,0.003,0.001],
                    'n_estimators': [10, 50, 100, 250]

                },
                "Gradient Boost": {
                    'learning_rate' : [1,0.3,0.1,0.03,0.01,0.003,0.001],
                    'criterion' : ['squared_error', 'friedman_mse'],
                    'n_estimators': [10, 50, 100, 250]
                },
                "Decision Tree" : {
                    'criterion' : ['squared_error', 'friedman_mse']
                },
                "Linear Regressor" : {},
                "XGB Regressor" : {
                    'learning_rate' : [1,0.3,0.1,0.03,0.01,0.003,0.001],
                    'n_estimators': [10, 50, 100, 250]

                },
                "CatBoost" : {
                    'depth' : [6,8,10],
                    'learning_rate' : [1,0.3,0.1,0.03,0.01,0.003,0.001],
                    'iterations' : [30,50,100]

                }, 
                "KNeighbours Regressor" : {
                    'n_neighbors' : [3,5,7,9,11]

                }



            }

            logging.info("Evaluating all the models...")

            model_report:dict = evaluate_model(X_train= X_train,y_train = y_train,X_test = X_test ,y_test = y_test, models = models,eval_metric=r2_score, params = params)

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




