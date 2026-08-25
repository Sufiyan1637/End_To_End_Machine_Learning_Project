import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

import os

class PredictPipeline:

    def __init__(self):
        pass

    # def predict(self, features):
    #     try:
    #         model_path = "artifacts/model.pkl"
    #         preprocessor_path = "artifacts/preprocesser.pkl"

    #         model = load_object(file_path=model_path)
    #         preprocessor = load_object(file_path=preprocessor_path)

    #         data_scaled = preprocessor.transform(features)

    #         pred = model.predict(data_scaled)

    #         return pred

    #     except Exception as e:
    #         raise CustomException(e, sys)


    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocesser.pkl"

            print("MODEL PATH:", os.path.abspath(model_path))

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("MODEL:", model)
            print("MODEL TYPE:", type(model))

            # IMPORTANT
            print("HAS COEF:", hasattr(model, "coef_"))

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:

    def __init__(self, gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):


        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score= writing_score


    def get_data_as_dataframe(self):

        try:
            custom_data = {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "reading_score" : [self.reading_score],
                "writing_score" : [self.writing_score] 
            }

            return pd.DataFrame(custom_data)

        except Exception as e :
            raise CustomException(e, sys)