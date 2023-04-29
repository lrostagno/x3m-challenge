# %%
import pandas as pd
from ast import literal_eval
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
# %%
def unnest_rows(df, column):
    # TODO: Add try catch
    df = df.explode(column)
    return df.join(pd.json_normalize(df[column])).drop(column, axis=1)

# %%
def replace_low_freq_values(df, column, freq_threshold):
    col_pct = df[column].value_counts(normalize=True) * 100
    low_freq_values = col_pct[col_pct < freq_threshold].index.tolist()
    df[column] = df[column].replace(low_freq_values, f'Other {column}')
    return df

# %%
def encode_binary_feature(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df
# %%
def plot_piechart(df, column):
    counts = df[column].value_counts()
    plt.pie(counts.values, labels=counts.index.values, autopct='%1.1f%%')
    plt.title(f'Distribution of {column}')
    plt.show()
    
def remove_outliers(df):
    # TODO: Parametrizar esta función
    df = df[df["latency"] <= 1000]
    df = df[df["ecpm"] <= 79]
    return df

def one_hot_encode(df, column):
    one_hot = pd.get_dummies(df[column])
    df = pd.concat([df, one_hot], axis=1)
    df = df.drop(column, axis=1)
    return df

# %%
def clean_data(df):
    
    df.waterfall_result = df.waterfall_result.apply(literal_eval)
    df = unnest_rows(df, "waterfall_result")
    df = df.drop(["event_id", "event_time", "app_id", "user_id", "id"], axis=1)
    df = remove_outliers(df)
    df = replace_low_freq_values(df, "country", 10)
    df = replace_low_freq_values(df, "connection_type", 3)
    df = replace_low_freq_values(df, "partner", 6)
    df = one_hot_encode(df, "adtype")
    df = one_hot_encode(df, "connection_type")
    df = one_hot_encode(df, "country")
    df = one_hot_encode(df, "partner")
    df = encode_binary_feature(df, "platform")
    return df


# %%
df = pd.read_csv('../Challege-Data.tsv', sep='\t')

# %%
df = clean_data(df)
# %%
df.iloc[0]["device"]
# %%
df.info()
# %%
len(df["user_id"].unique())
# %%
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df.describe().T.round(3)

# %%


# %%
columns = ['partner']
for column in columns:
    plot_piechart(df, column)

# %%
df["event_time"].apply(lambda x : str(x)[:13]).unique()
# %%
df.columns
# %%
df["app_id"].unique()

# %%
df.corr()
# %%

df["error"].unique()


# %%

df.head()
# %%
def train_and_test_model(df):
    df = df.drop(["device", "partner", "error", "auction_id"], axis=1)
    X = df.drop("latency", axis=1)
    y = df["latency"]
    scaler = StandardScaler()
    X[["ecpm"]] = scaler.fit_transform(X[["ecpm"]])
    X_train, X_test, y_train, y_test = train_test_split(
                                            X, 
                                            y, 
                                            test_size=0.01, 
                                            random_state=42)
    X_train = X_train["ecpm"].values.reshape(-1, 1)
    X_test = X_test["ecpm"].values.reshape(-1, 1)
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    return lin_reg.score(X_test, y_test)
    



# %%
train_and_test_model(df)
# %%
predictions
# %%
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(lin_reg, f)
# %%
df.columns
# %%

import pandas as pd

# create a sample DataFrame with an array of dictionaries
data = {'id': [1, 2, 3], 
        'name': ['Alice', 'Bob', 'Charlie'], 
        'scores': [[{'math': 90, 'english': 80}, {'math': 85, 'english': 95}], 
                   [{'math': 75, 'english': 90}], 
                   [{'math': 80, 'english': 85}, {'math': 95, 'english': 70}, {'math': 90, 'english': 80}]]}

df = pd.DataFrame(data)
# %%


# %%
df.columns
# %%



# %%



# %%
# %%
# Add the one-hot encoded columns to the original dataframe

df = one_hot_encode(df, "adtype")
# %%
df
# %%
df
# %%
df.drop("ban", axis=1)

# %%
