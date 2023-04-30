from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
# %%
# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# %%
# Define a route to accept POST requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the POST request
    test = request.json['test']

    # Convert the features to a numpy array
    #features = np.array(features).reshape(1, -1)

    # Make a prediction using the model
    #prediction = model.predict(features)

    # Return the prediction as a JSON response
    # response = {'prediction': float(prediction)}
    response = {'prediction' : test}
    return jsonify(response)

if __name__ == '__main__':
    # Run the app on port 5000
    app.run(port=5000, debug=True)
