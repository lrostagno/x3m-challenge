# %%
import requests
import json
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder
import pickle
import xgboost as xgb
# from EDA import clean_data


# %%
def set_categorical_features(row, categorical_features_dict):
    for column, categories in categorical_features_dict.items():
        if row[column] in categories:
            row[f"{column},{row[column]}"] = 1
        else:
            row[f"{column},other_{column}"] = 1
    return row

def split_array_to_dict(arr):
    d = {}
    [d.setdefault(k, []).append(v) for k, v in [elem.split(',',1) for elem in arr if len(elem.split(',', 1)) > 1]]
    return d


def unnest_rows(df, column, explode=False):
    df[column] = df[column].apply(literal_eval)
    # TODO: Add try catch
    if explode:
        df = df.explode(column)
    df.reset_index(inplace=True)
    df = pd.concat([df, pd.json_normalize(df[column])], axis=1)
    df = df.drop([column], axis=1)
    return df


def one_hot_encode(df, column):
    one_hot = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, one_hot], axis=1)
    df = df.drop(column, axis=1)
    return df

def encode_binary_feature(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df
# %%
path = '../Challege-Data.tsv'
with open('model_API.pkl', 'rb') as f:
    model = pickle.load(f)
df = pd.read_csv(path, sep='\t')

json_data = df.to_json(orient='records')
import requests
api_endpoint = 'http://localhost:5000/predict'
headers = {'Content-Type': 'application/json'}
# %%
response = requests.post(api_endpoint, headers=headers, data=json_data)
# %%
if response.status_code == 200:
    print(response.json())
else:
    print('Error:', response.status_code)
def clean_data_inference(df):
    df = unnest_rows(df, "waterfall_result", explode=True)
    df = unnest_rows(df, "device")
    df["area"] = df["w"] * df["h"]
    df = df.drop(["id", "event_id","level_0", "index", "event_time",
                    "user_id", "auction_id", "model", "hwv", "error", "w", "h", 
                    "memory_total"], axis=1)

    categorical_features_dict = split_array_to_dict(model.feature_names_in_)
    cols_to_add = set(model.feature_names_in_).difference(set(df.columns))
    cols_to_remove = set(df.columns).difference(set(model.feature_names_in_))
    df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=cols_to_add)], axis=1)
    df = df.copy()
    df = df.apply(set_categorical_features, axis=1, args=(categorical_features_dict,))
    df = df.drop(cols_to_remove, axis=1)
    if len(df) != len(df.dropna()):
        print("Advertencia... algunos datos están incompletos y serán eliminados")
    df = df.dropna()
    return df


# %%
df_cleaned = clean_data_inference(df.head(100))

# %%
import pandas as pd
import tempfile
import os





    

# %%


# %%
model.predict(json.loads(x))
# %%

# %%
import numpy as np
json_data = x

data = json.loads(json_data)

values = np.array([list(data.values())])


prediction = model.predict(values)
# %%
x
# %%
X_test = pd.DataFrame.from_dict(df)
# %%
# %%

# %%
x = df.head(10).to_json(orient='records')

# %%
df_test = pd.DataFrame.from_dict(json.loads(x))
# %%
df_test = pd.DataFrame.from_dict(json.loads(x))

# %%
df_test.drop_index()

# %%
