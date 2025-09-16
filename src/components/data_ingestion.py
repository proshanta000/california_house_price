import os
import sys

# Get the absolute path of the current script's directory.
# This ensures that no matter where you run the script from,
# it can always find the project's root directory.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Move up two levels to reach the project root directory
# For example: E:/Data_science/Projects/california_house_price
project_root = os.path.join(script_dir, '..', '..')

# Add the project root to the system path
# This allows Python to find and import modules from the 'src' directory
sys.path.append(project_root)

# Now, your imports will work correctly.
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTraningConfig, ModelTrainer

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


## Intialize the Data Ingestion Configuration

@dataclass
class DataIngestionConfig:
    # Configuration for data ingestion paths
    def __init__(self):
        self.train_data_path = os.path.join('artifacts', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'test.csv')
        self.raw_data_path = os.path.join('artifacts', 'raw_data.csv')


class DataIngestion:
    # Initializing the data ingestion configaration
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    #Initiating the data ingestion process
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component.")

        try:
            housing = fetch_california_housing()
            df = pd.DataFrame(housing.data, columns=housing.feature_names)
            df['Price'] =housing.target
            logging.info("Dataset read as pandas dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

        
            logging.info("Train Teat and split.")
            train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is compleated")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Exception occured at Data Ingestion stage")
            raise CustomException(e, sys)
        
if __name__=="__main__":
    # The main execution block of the pipeline
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)


        model_trainer = ModelTrainer()
        # Capture the returned values from the model trainer
        best_model_name, best_model_score, best_model = model_trainer.initiate_model_traning(train_arr, test_arr)
        
        # Now we can print the actual results
        print(f"\nModel Training Complete.")
        print(f"Best Model Found: {best_model_name}")
        print(f"R2 Score: {best_model_score:.4f}\n")

    except Exception as e:
        logging.info("An exception occurred during the pipeline execution.")
        raise CustomException(e, sys)
