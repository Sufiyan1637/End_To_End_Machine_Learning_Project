import sys
import pandas as pd
import numpy as np
import os
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path , obj):
    try:
        dir_name = os.path.dirname(file_path)

        os.makedirs(dir_name, exist_ok=True)

        with open(file_path , "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)



def evaluate_model(x_train, y_train, x_test, y_test, models, params):

    report = {}
    best_models = {}

    for i in range(len(list(models))):

        model_name = list(models.keys())[i]
        model = list(models.values())[i]
        param = list(params.values())[i]

        gc = GridSearchCV(
            model,
            param_grid=param,
            cv=5,
            n_jobs=-1
        )

        gc.fit(x_train, y_train)

        best_model = gc.best_estimator_

        y_train_pred = best_model.predict(x_train)
        y_test_pred = best_model.predict(x_test)

        train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)

        report[model_name] = test_model_score
        best_models[model_name] = best_model

        print(
            model_name,
            "Train Score:", train_model_score,
            "Test Score:", test_model_score
        )

    return report, best_models


def load_object(file_path):

    try:
        with open(file_path, "rb") as file_obj:

            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)