import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts", "model.pkl")



class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    try:

        def initiate_model_trainer(self, train_array, test_array):

            x_train, x_test, y_train, y_test = (
                train_array[: , :-1],
                test_array[: , :-1],
                train_array[: , -1],
                test_array[:, -1]
            ) 


            models = {
                        "Random Forest" : RandomForestRegressor(),
                        "Decision Tree" : DecisionTreeRegressor(),
                        "Gradient Boosting" : GradientBoostingRegressor(),
                        "Linear Regression" : LinearRegression(),
                        "K-Neighbours" : KNeighborsRegressor(),
                        "XGBRegressor" : XGBRegressor(),
                        "Ada Boost" : AdaBoostRegressor(),
                    }
            
            
            params = {
                "Random Forest" : 
                {
                    "criterion" : ["squared_error"  , "absolute_error", "poisson"],
                    "n_estimators" : [8,16,34,64,128,256],
                    "max_features" : ["sqrt" , "log2" , None]
                    
                },
                "Decision Tree" :
                {
                    "criterion" : ["squared_error"  , "absolute_error", "poisson"],
                    "splitter" : ["best" , "random"],
                    "max_features" : ["sqrt" , "log2"]
                },


                "Gradient Boosting" : 
                {
                    "loss" : ["squared_error" , "huber" , "absolute_error", "quantile"],
                    "learning_rate" : [.1 , .01 , .001 , .05],
                    "subsample" : [.6 , .7 , .75 , .80 , .85 , .90],
                    "criterion" : ["squared_error" ],
                    "max_features" : [None , "sqrt" , "log2"],
                    "n_estimators" : [8,16,32,64,128,256]
                },

                "Linear Regression" : {},

                "K-Neighbours" :
                {
                    "weights" : ['uniform' , 'distance'],
                    "algorithm" : ['ball_tree' , 'kd_tree' , 'brute']
                },

                "XGBRegressor" : 
                {
                    'learning_rate' : [.1 , .01 , .001 , .05],
                    'n_estimators' : [8,16,32,64,128,256]
                },

                "Ada Boost" : 
                {
                    'learning_rate' : [.1 , .01 , .001 , .05],
                    'n_estimators' : [8,16,32,64,128,256]
                    
                }
            
            }

            model_report:dict = evaluate_model(x_train=x_train, x_test=x_test,y_train=y_train, y_test=y_test , models=models, params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            save_object(self.model_trainer_config.model_file_path, best_model)

            print(best_model)


    except Exception as e:

        raise CustomException(e,sys)