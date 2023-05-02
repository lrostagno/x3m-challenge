import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from ast import literal_eval
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
    cols_to_drop = list(set(["level_0", "index", "event_time",
                    "user_id", "auction_id", "model", "hwv", "error", "w", "h", 
                    "memory_total"]).intersection(set(df.columns)))
    df = df.drop(cols_to_drop, axis=1)

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


def unnest_rows(df, column, explode=False):
    df[column] = df[column].apply(literal_eval)
    if explode:
        df = df.explode(column)
    df.reset_index(inplace=True)
    df = pd.concat([df, pd.json_normalize(df[column])], axis=1)
    df = df.drop([column], axis=1)
    return df


def replace_low_freq_values(df, column, freq_threshold):
    col_pct = df[column].value_counts(normalize=True) * 100
    low_freq_values = col_pct[col_pct < freq_threshold].index.tolist()
    df[column] = df[column].replace(low_freq_values, f'other_{column}')
    return df


def encode_binary_feature(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df

    
def remove_outliers(df, column, threshold):
    df = df[df[column] <= threshold]
    return df

def one_hot_encode(df, column):
    one_hot = pd.get_dummies(df[column], prefix=column, prefix_sep=",")
    df = pd.concat([df, one_hot], axis=1)
    df = df.drop(column, axis=1)
    return df

def process_categorical_features(df):
    df = replace_low_freq_values(df, "country", 10)
    df = replace_low_freq_values(df, "connection_type", 10)
    df = replace_low_freq_values(df, "partner", 10)
    df = replace_low_freq_values(df, "app_id", 10)
    df = one_hot_encode(df, "adtype")
    df = one_hot_encode(df, "connection_type")
    df = one_hot_encode(df, "country")
    df = one_hot_encode(df, "partner")
    df = one_hot_encode(df, "app_id")
    df = one_hot_encode(df, "platform")
    df = one_hot_encode(df, "type")
    return df


def download_and_clean_data(path):
    df = pd.read_csv(path, sep='\t')
    df = unnest_rows(df, "waterfall_result", explode=True)
    df = unnest_rows(df, "device")
    df["area"] = df["w"] * df["h"]
    df = df.drop(["id", "event_id","level_0", "index", "event_time",
                  "user_id", "auction_id", "model", "hwv", "error", "w", "h", "ppi",
                  "memory_total"], axis=1)
    df = remove_outliers(df, "latency", 1000)
    df = process_categorical_features(df)
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse:.2f}")
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.5f}")
    print(f"Mean Absolute Error: {mae:.5f}")
    print(f"R-squared: {r2:.5f}")