import os
import sys 
import pandas as pd
import numpy as np 

from dataclasses import dataclass

# Get the absolute path of the current script's directory.
# This ensures that no matter where you run the script from,
# it can always find the project's root directory.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move up one level to reach the project root directory.
# For example: E:/Data_science/Projects/california_house_price
project_root = os.path.join(script_dir, '..')

# Add the project root to the system path
# This allows Python to find and import modules from the 'src' directory.
sys.path.append(project_root)

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor, 
    GradientBoostingRegressor, 
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evalute_models


@dataclass
class ModelTraningConfig:
    """
    Configuration for model training path and parameters
    """
    traning_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    """
    Class for training machine learning models
    """
    def __init__(self):
        self.model_traning_config = ModelTraningConfig()
    
    def initiate_model_traning(self, train_array, test_array):
        """
        This method trains multiple regression models and evaluates their performance.
        It returns the best model based on r2 score.
        """
        try:
            logging.info("Entering the model training method or component.")

            # Fix for the CatBoostError: create an absolute path for the writable directory
            catboost_dir = os.path.join(os.getcwd(), 'artifacts', 'catboost_info')
            os.makedirs(catboost_dir, exist_ok=True)
            logging.info(f"Created directory for CatBoost: {catboost_dir}")

            logging.info("Splitting the training and testing input and target variables.")
            # Splitting the train and test arrays into features and target variables
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            #Creating models dictionary with various regression models
            models = {
                'Linear Regression': LinearRegression(),
                'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
                'Random Forest Regressor': RandomForestRegressor(random_state=42),
                'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
                'XGB Regressor': XGBRegressor(random_state=42),
                'KNeighbors Regressor': KNeighborsRegressor(),
                # Fix for the CatBoostError: specify a writable directory
                'CatBoost Regressor': CatBoostRegressor(verbose=False, random_state=42, train_dir=catboost_dir),
                'AdaBoost Regressor': AdaBoostRegressor(random_state=42)
            }

            # Define parameters grid for each model
            # IMPORTANT: Adjust these parameters and ranges based on dataset and computational resources.
            # Start with broader ranges and narrow down as results come.
            params = {
                'Linear Regression': {}, # No hyperparameters for basic Linear Regression in this context
                # Corrected key name to match the models dictionary
                'Decision Tree Regressor': {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    'max_depth': [10, 20, None], # Adjusted range for faster execution
                    'min_samples_split': [2, 5], # Adjusted range
                    'min_samples_leaf': [1, 2] # Adjusted range
                },
                'Random Forest Regressor': {
                    'n_estimators': [64, 128, 256], # Adjusted range
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [10, 20, None], # Adjusted range
                    'min_samples_split': [2, 5], # Adjusted range
                    'min_samples_leaf': [1, 2] # Adjusted range
                },
                'Gradient Boosting Regressor': {
                    'learning_rate': [.1, .01, .05], # Adjusted range
                    'subsample': [0.7, 0.8, 0.9], # Adjusted range
                    'n_estimators': [64, 128, 256], # Adjusted range
                    'max_depth': [3, 5] # Adjusted range
                },
                'XGB Regressor': {
                    'learning_rate': [.1, .01, .05], # Adjusted range
                    'n_estimators': [64, 128, 256], # Adjusted range
                    'max_depth': [3, 5], # Adjusted range
                    'subsample': [0.7, 0.8], # Adjusted range
                    'colsample_bytree': [0.7, 0.8], # Adjusted range
                    'gamma': [0, 0.1] # Adjusted range
                },
                'KNeighbors Regressor': {
                    'n_neighbors': [5, 7, 9], # Adjusted range
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['ball_tree', 'kd_tree'] # Adjusted range
                },
                'CatBoost Regressor': {
                    'depth': [6, 8], # Adjusted range
                    'learning_rate': [0.01, 0.05], # Adjusted range
                    'iterations': [50, 100] # Adjusted range
                },
                'AdaBoost Regressor': {
                    'learning_rate': [.1, .01, 0.5], # Adjusted range
                    'n_estimators': [64, 128, 256] # Adjusted range
                }
            }
            
            # Call the updated evalute models function
            model_reports, tuned_models = evalute_models(
                X_train= X_train, 
                y_train= y_train,
                X_test= X_test,
                y_test= y_test,
                models= models,
                params= params
            )
            
            # Finding the best model based on r2 score from the models report dictionary
            best_model_score = max(sorted(model_reports.values()))

            # Finding the best model name from model report dictionary 
            best_model_name = list(model_reports.keys())[
                list(model_reports.values()).index(best_model_score)
            ]

            # Retrieve the actual best tuned model instance
            best_model = tuned_models[best_model_name]

            # If the best model score is less than 0.6, raise an exception
            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy.", sys)

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score:.4f}")

            save_object(
                file_path=self.model_traning_config.traning_model_file_path,
                obj=best_model
            )

            # This is the new line that returns the best model and its score
            return best_model_name, best_model_score, best_model
            
        except Exception as e:
            logging.info("Exception occurred in the initiate_model_training.")
            raise CustomException(e, sys)
