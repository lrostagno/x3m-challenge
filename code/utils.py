import requests
import json
import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder
import pickle
import xgboost as xgb

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



def clean_data_inference(df, model):
    df = unnest_rows(df, "waterfall_result", explode=True)
    df = unnest_rows(df, "device")
    df["area"] = df["w"] * df["h"]
    df = df.drop(["level_0", "index", "event_time",
                    "user_id", "auction_id", "model", "hwv", "error", "w", "h", 
                    "memory_total"], axis=1)

    categorical_features_dict = split_array_to_dict(model.feature_names_in_)
    cols_to_add = set(model.feature_names_in_).difference(set(df.columns))
    cols_to_remove = set(df.columns).difference(set(model.feature_names_in_).
                                                union(set(["id", "event_id"])))
    df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=cols_to_add)], axis=1)
    df = df.apply(set_categorical_features, axis=1, args=(categorical_features_dict,))
    df = df.drop(cols_to_remove, axis=1)
    if len(df) != len(df.dropna()):
        print("Advertencia... algunos datos están incompletos y serán eliminados")
    df = df.dropna()
    df = df.rename(columns={'id': 'instance_id'})
    return df