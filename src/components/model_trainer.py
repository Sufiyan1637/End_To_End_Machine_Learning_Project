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
                "Ada Boost" : AdaBoostRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "K Neighbors" : KNeighborsRegressor(),
                "Linear Regression" : LinearRegression(),
                "XG BOOST" : XGBRegressor()
            }

            model_report:dict = evaluate_model(x_train=x_train, x_test=x_test,y_train=y_train, y_test=y_test , models=models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(models.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            save_object(self.model_trainer_config.model_file_path, best_model)

            print(best_model)


    except Exception as e:

        raise CustomException(e,sys)