from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from io import BytesIO

app = Flask(__name__)


model = pickle.load(open('model.pkl', 'rb'))


# Define a route to accept POST requests
@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the POST request
    data = request.get_json()
    df = pd.DataFrame.from_dict(data)

    response = {'latency' : df.iloc[0]["event_id"]}
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    # Run the app on port 5000
    app.run(port=5000, debug=True)



def raw_prediction():
    pass