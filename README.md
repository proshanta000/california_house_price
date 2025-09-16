### End to end mechin learning project about calefornia house project


### California House Price Prediction
```text
This project is an end-to-end machine learning pipeline designed to predict the median house value in California districts based on the 1990 California census data. The pipeline is built using modular components for data ingestion, data transformation, model training, and prediction, following best practices for MLOps.
```

#### Features
**Data Ingestion:** Automatically downloads the California housing dataset from scikit-learn and splits it into training and testing sets.

**Data Transformation:** Uses a custom preprocessor to scale numerical features, saving the preprocessor object for future use in the prediction pipeline.

**Model Training:** Trains multiple regression models (including Linear Regression, XGBoost, CatBoost, etc.) and evaluates their performance to find the best-performing model.

**Model Training:** Saves the trained model as a pickle file for easy deployment and prediction.

**Prediction Pipeline:** Provides a robust and reusable pipeline for making predictions on new data points.

#### Project Structure
The project is organized into the following directories:
```text
/california_house_pricing
|-- artifacts/                 # Stores all pipeline outputs (data, models, preprocessors)
|   |-- train.csv
|   |-- test.csv
|   |-- raw_data.csv
|   |-- preprocessor.pkl
|   |-- model.pkl
|-- src/                       # Source code for the ML pipeline components
|   |-- components/
|   |   |-- data_ingestion.py      # Handles data loading and splitting
|   |   |-- data_transformation.py # Handles data preprocessing
|   |   |-- model_trainer.py       # Handles model training and evaluation
|   |-- pipline/
|   |   |-- predict_pipline.py     # Main prediction logic
|   |-- __init__.py
|   |-- exception.py             # Custom exception handling module
|   |-- logger.py                # Custom logging module
|   |-- utils.py                 # Utility functions (e.g., for saving objects)
|-- app.py                     # Flask application to serve the prediction model
|-- templates/                 # HTML templates for the web application
|   |-- index.html
|   |-- predict_datapoint.html
|   |-- result.html
|-- requirements.txt           # Project dependencies
|-- setup.py                   # For package installation
|-- README.md                  # This file

```

### Getting Started
#### Prerequisites
1. [Github Account](https:/github.com)
2. [Heroku Account](https://www.heroku.com/)
3. [Vs code IDE](https://code.visualstudio.com/)
4. [GitCLI](https://git-scm.com/)


> Create a virtual environment and activate it:
```bash
    conda create -p venv python==3.13
```

> Install the required dependencies:

```bash
python setup.py
```
### Usage
1. Running the ML Pipeline
To execute the entire data ingestion, transformation, and model training pipeline, you can run the main script. This will generate the necessary files in the artifacts/ directory.
```bash
python src/components/data_ingestion.py
```

2. Running the Web Application
After the pipeline has created the preprocessor.pkl and model.pkl files, you can start the Flask web application to make predictions.
```python
python app.py
```

The application will run on http://127.0.0.1:5000. Open this URL in your browser to access the prediction form.

#### Contributing
Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

#### License
Apache License
Version 2.0, January 2004

#### Contact
For any questions, feel free to contact me at your-email@example.com.

