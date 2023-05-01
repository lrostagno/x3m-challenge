from flask import Flask, request, jsonify
import pickle
import pandas as pd
from utils import *

app = Flask(__name__)
model = pickle.load(open('/home/leandro/Documents/OTHERS/ML/x3m-challenge/code/model_API.pkl', 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame.from_dict(data)
    df_predictions = clean_data_inference(df.head(3), model)
    predictions = model.predict(df_predictions[model.feature_names_in_]).tolist()
    predictions = [int(x) for x in predictions]
    df_predictions["latency"] = predictions

    final_output = df_predictions[["event_id", "instance_id", "latency"]].to_json(orient="records")
    response = {'predictions' : final_output}
    print(response)
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

