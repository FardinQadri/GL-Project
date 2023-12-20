from flask import Flask, jsonify
import joblib
import numpy as np
import requests

app = Flask(__name__)

# Load the model
my_rf_model = joblib.load('my_model.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data points from the request (assuming JSON input)
    data = requests.get_json()

    # Ensure there are 8 data points
    if len(data) != 8:
        return jsonify({'error': 'Expected 8 data points'})

    # Convert the data to a format that the model expects (list to numpy array)
    data_array = np.array(data).reshape(1, -1)  # Reshape to a 2D array

    # Make predictions
    prediction = my_rf_model.predict(data_array)

    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
