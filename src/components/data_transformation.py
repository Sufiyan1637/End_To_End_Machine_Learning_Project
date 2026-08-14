import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path = os.path.join("artifacts", "preprocesser.pkl")



class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        '''
            This Function is Responsible for Data Transformation
        '''
        try:
            numerical_feature = ["writing_score", "reading_score"]
            categorical_feature = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            numerical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler" , StandardScaler())
            ])

            logging.info("encoding of Numerical Feature is Completed")

            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe" , OneHotEncoder(handle_unknown="ignore")),
                ("scaler" , StandardScaler(with_mean=True))
            ])

            logging.info("Encoding of Categorical Feature is Completed")

            preprocesser = ColumnTransformer(transformers=[
                ("numerical_transformer" , numerical_pipeline , numerical_feature),
                ("categorical_transformer" , categorical_pipeline , categorical_feature)

            ])


            logging.info("Transformation of Both Numerical and Categorical are Completed")


            return preprocesser
        
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_tranformation(self, train_path , test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test Data")

            logging.info("Obtaining Preprocesser Object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = "math_score"
            numerical_feature = ["writing_score", "reading_score"]


            input_train_feature = train_df.drop(columns=[target_column])
            target_train_feature = train_df[target_column]

            input_test_feature = test_df.drop(columns=[target_column])
            target_test_feature = test_df[target_column]


            logging.info("Applying Preprocesser Object on Training  and Testing Dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_train_feature)
            input_feature_test_arr = preprocessing_obj.fit_transform(input_test_feature)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_train_feature)
            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_test_feature)]

            logging.info("Saved Preprocessing Object.")

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)