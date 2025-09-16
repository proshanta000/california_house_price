from flask import Flask, request, render_template
import numpy as np
import pandas as pd

# from sklearn.preprocessing import StandardScaler
from src.pipline.predict_pipline import PreddictPipline, CustomData


application = Flask(__name__)

app = application

# Route for the home page
@app.route('/')
def home():
    """
    Renders the main input form page (index.html).
    """
    return render_template('index.html')

# Route to handle both displaying the form and processing the prediction
@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handles displaying the prediction form (GET) and processing
    the form submission (POST) to make a prediction.
    """
    if request.method == 'GET':
        # If the request is a GET, render the form page
        return render_template('predict_datapoint.html')
    else:
        # If the request is a POST (form submission), process the data
        # Create a CustomData object from form inputs
        data = CustomData(
            MedInc=float(request.form.get('MedInc')),
            HouseAge=float(request.form.get('HouseAge')),
            AveRooms=float(request.form.get('AveRooms')),
            AveBedrms=float(request.form.get('AveBedrms')),
            Population=float(request.form.get('Population')),
            AveOccup=float(request.form.get('AveOccup'))
        )
        
        # Convert CustomData to a DataFrame for the prediction pipeline
        pred_df = data.get_data_as_dataframe()
        print(pred_df) # For debugging

        # Initialize and run your prediction pipeline
        predict_pipeline = PreddictPipline()
        results = predict_pipeline.predict(pred_df)
        
        # Render the result page, passing the prediction result
        return render_template('result.html', results=results[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
