import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PreddictPipline:
    """
    class for making prediction  using a trained model
    """
    def __init__(self):
        """self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model = load_object(self.model_path)
        self.preprocessor = load_object(self.preprocessor_path)"""
        pass
    
    def predict(self, feature):
        model_path = 'artifacts/model.pkl'
        preprocessor_path = 'artifacts/preprocessor.pkl'
        model = load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)

        try:
            #preprocess the input features
            data_scaled= preprocessor.transform(feature)
            logging.info('Data Preprocessing conpleted')

            # Make prediction using the model

            prediction = model.predict(data_scaled)
            logging.info("prediction completed")

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    """
    Class for handling custom data input for prediction .
    """
    def __init__(self,
                 MedInc : float,
                 HouseAge : float,
                 AveRooms : float,
                 AveBedrms : float,
                 Population : float,
                 AveOccup : float
                 ):

        self.MedInc = MedInc
        self.HouseAge = HouseAge
        self.AveRooms = AveRooms
        self.AveBedrms = AveBedrms 
        self.Population = Population
        self.AveOccup = AveOccup
    
    def get_data_as_dataframe(self):
        """
        Converts the custom data instance into a pandas DataFrame.
        
        This is a crucial step as most machine learning models are trained
        to accept input in a DataFrame format.
        
        Returns:
            pd.DataFrame: A DataFrame containing the custom data.
        """
        try:
            custom_data_input_dict = {
                "MedInc": [self.MedInc],
                "HouseAge": [self.HouseAge],
                "AveRooms": [self.AveRooms],
                "AveBedrms": [self.AveBedrms],
                "Population": [self.Population],
                "AveOccup": [self.AveOccup]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            # Raise a custom exception if there is an error in data conversion
            logging.error("Error occurred while creating DataFrame from custom data.")
            raise CustomException(e, sys)
