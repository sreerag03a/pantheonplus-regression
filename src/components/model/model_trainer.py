import numpy as np
import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,BayesianRidge
from sklearn.metrics import r2_score, mean_absolute_error,root_mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.components.handling.exceptions import CustomException
from src.components.handling.logger import logging
from src.components.handling.utils import save_obj, evaluate_model,train_model


@dataclass
class ModelConfig:
    models_config = os.path.join('outputs','models')
    trained_model_path = os.path.join('outputs','models','trained_model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_config = ModelConfig()
    
    '''
    Training of models and hyperparameter tuning to find the best performing model.

    The best model found is pickled by default - We can tune all models and pickle them as well - Both are done in project_stage.py.
    '''
    def start_trainer(self,train_set,test_set, train_all = False):
        os.makedirs(self.model_config.models_config, exist_ok=True)
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
        models_path = os.path.join(self.model_config.models_config)
        try:
            logging.info("Initializing train and test sets")
            X_train,y_train,X_test,y_test = (
                train_set[:,:-1],
                train_set[:,-1],
                test_set[:,:-1],
                test_set[:,-1]
            )
        except Exception as e:
            raise CustomException(e,sys)
        if train_all == True:
            try:
                for i in range(len(list(models))):
                    logging.info('Evaluating {}'.format(list(models.keys())[i]))
                    model = list(models.values())[i]

                    param = params[list(models.keys())[i]]

                    gs = GridSearchCV(model, param, cv = 3)

                    gs.fit(X_train, y_train)


                    model.set_params(**gs.best_params_)
                    model.fit(X_train,y_train)
    
                    model_path = os.path.join(models_path,f'{list(models.keys())[i]}.pkl')
                    logging.info('Completed Hyperparameter tuning of model {}.'.format(list(models.keys())[i]))
                    save_obj(
                    filepath= model_path,
                    obj=model
                    )
                    logging.info('Saved model in {}'.format(model_path))
                
                
                return 'completed'

            except Exception as e:
                raise CustomException(e,sys)

        elif train_all == False:
            try:

                logging.info("Evaluating all the models...")

                model_report:dict = evaluate_model(X_train= X_train,y_train = y_train,X_test = X_test ,y_test = y_test, models = models,eval_metric=r2_score, params = params)

                logging.info("Model evaluation complete. Determining best model...")
                best_score = max(sorted(model_report.values()))

                if best_score < 0.6 :
                    raise CustomException("No model is good enough for this data.")
                
                logging.info("Best model found.")


                best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]

                logging.info("Best model is {} with score {}, saving it to {}".format(best_model_name, best_score, self.model_config.trained_model_path))
                best_model = models[best_model_name]
                save_obj(
                    filepath= self.model_config.trained_model_path,
                    obj=best_model

                )
                return best_model_name



            except Exception as e:
                raise CustomException(e,sys)


@dataclass
class AdvancedModelConfig:
    models_config = os.path.join('outputs','models','advanced_models')
    trained_model_path = os.path.join('outputs','models','advanced_models','trained_advanced_model.pkl')


class AdvancedModelTrainer:
    '''
    Training for advanced models incorporating uncertainty approximation techniques. Models are trained and saved separately.
    Since we are passing in the measurement errors for this evaluation, we won't be using Grid Search Cross Validation to tune hyperparameters.
    
    
    ## Maybe Implement a self made solution in the future

    '''

    def __init__(self):
        self.model_config = AdvancedModelConfig()
        self.kernel = RBF(length_scale=1.0)

    def start_trainer(self,train_set,test_set,covmatrix=0,train_all=False):
        os.makedirs(self.model_config.models_config, exist_ok=True)
        advanced_models={"Gaussian Process Regressor" : GaussianProcessRegressor(
                        kernel=self.kernel,copy_X_train=True,normalize_y=True,alpha=covmatrix),
                         "Bayesian Ridge Regressior": BayesianRidge(copy_X=True)}
        models_path = os.path.join(self.model_config.models_config)
        try:
            logging.info("Initializing train and test sets")
            X_train,y_train,X_test,y_test = (
                train_set[:,:-1],
                train_set[:,-1],
                test_set[:,:-1],
                test_set[:,-1]
            )
            logging.info(f'{X_train.shape},{y_train.shape}')
        except Exception as e:
            raise CustomException(e,sys)
        if train_all == True:
            try:
                for i in range(len(list(advanced_models))):
                    logging.info('Evaluating {}'.format(list(advanced_models.keys())[i]))
                    model = list(advanced_models.values())[i]
                    model.fit(X_train,y_train)
    
                    model_path = os.path.join(models_path,f'{list(advanced_models.keys())[i]}.pkl')
                    logging.info('Completed Hyperparameter tuning of model {}.'.format(list(advanced_models.keys())[i]))
                    save_obj(
                    filepath= model_path,
                    obj=model
                    )
                    logging.info('Saved model in {}'.format(model_path))
                
                
                return 'completed'

            except Exception as e:
                raise CustomException(e,sys)
        elif train_all == False:
            try:

                logging.info("Evaluating all the models...")

                model_report:dict = train_model(X_train= X_train,y_train = y_train,X_test = X_test ,y_test = y_test, models = advanced_models,eval_metric=r2_score)

                logging.info("Model evaluation complete. Determining best model...")
                best_score = max(sorted(model_report.values()))

                if best_score < 0.6 :
                    raise CustomException("No model is good enough for this data.")
                
                logging.info("Best model found.")


                best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]

                logging.info("Best model is {} with score {}, saving it to {}".format(best_model_name, best_score, self.model_config.trained_model_path))
                best_model = advanced_models[best_model_name]
                save_obj(
                    filepath= self.model_config.trained_model_path,
                    obj=best_model

                )
                return best_model_name



            except Exception as e:
                raise CustomException(e,sys)