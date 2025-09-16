import os 
import sys
import pandas as pd 
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data transformation initiated.")
            
            # The columns to apply standardization on
            # These must be the columns that exist in the DataFrame after dropping the target and other columns.
            features_to_scale = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

            ## Pipeline for scaling
            scaler_pipeline = Pipeline(
                steps=[
                    ('scaler', StandardScaler())
                ]
            )

            # Create a ColumnTransformer to apply the scaling pipeline
            preprocessor = ColumnTransformer([
                ('scaler_pipeline', scaler_pipeline, features_to_scale)
            ])
            
            logging.info('Pipeline for data transformation completed.')
            return preprocessor

        except Exception as e:
            logging.info("Error in Data transformation.")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data complete.")
            logging.info(f"Train DataFrame Head: \n{train_df.head().to_string()}")
            logging.info(f"Test DataFrame Head: \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessing object.")
            preprocessing_object = self.get_data_transformation_object()

            target_column_name = "Price"
            # Drop columns that are not features for the model
            drop_column_name = [target_column_name, 'Latitude', 'Longitude']

            input_feature_train_df = train_df.drop(columns=drop_column_name, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_column_name, axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on train and test data.")

            # Transform the feature data
            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_feature_test_df)

            # Combine transformed features with the target column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )

            logging.info('Preprocessor pickle file saved.')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info('Exception occurred in the initiate_data_transformation.')
            raise CustomException(e, sys)
            

