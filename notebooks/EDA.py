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
import xgboost as xgb
from sklearn.metrics import mean_squared_error
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
def clean_data():
    df = pd.read_csv('../Challege-Data.tsv', sep='\t')
    df["waterfall_result"] = df["waterfall_result"].apply(literal_eval)
    df = unnest_rows(df, "waterfall_result")
    df = df.drop(["event_id", "event_time", "app_id", "user_id", "id"], axis=1)
    df = remove_outliers(df)
    df = replace_low_freq_values(df, "country", 10)
    df = replace_low_freq_values(df, "connection_type", 3)
    df = replace_low_freq_values(df, "partner", 1)
    df = one_hot_encode(df, "adtype")
    df = one_hot_encode(df, "connection_type")
    df = one_hot_encode(df, "country")
    df = one_hot_encode(df, "partner")
    df = encode_binary_feature(df, "platform")
    df["device"] = df["device"].apply(literal_eval)
    df = df.join(pd.json_normalize(df["device"])).drop("device", axis=1)
    df = one_hot_encode(df, "type")
    scaler = StandardScaler()
    df[["ecpm", "w", "h", "memory_total","ppi"]] = scaler.fit_transform(df[["ecpm", "w", "h", "memory_total", "ppi"]])
    return df


# %%
df.describe().T

# %%
df = clean_data()
# %%
df.corr()
# %%
# %%
df["ppi"].unique()[4]
# %%
# %%
# %%
df_test = df.join(pd.json_normalize(df["device"])).drop("device", axis=1)
# %%
df_test["type"]
# %%
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df.describe(percentiles=[[.25, .5, .75, .99]]).T.round(3)

# %%
len(df.columns)

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

df["partner"].value_counts()


# %%
df.columns
# %%

df.head()
# %%
def train_and_test_model(df):
    #df = df.drop(['platform', 'country', 'adtype', 'connection_type', 'device','error', 'auction_id',], axis=1)
    df = df.drop([ 'device','error'], axis=1)
    X = df.drop("latency", axis=1)
    y = df["latency"]

    X_train, X_test, y_train, y_test = train_test_split(
                                            X, 
                                            y, 
                                            test_size=0.01, 
                                            random_state=42)
    X_train = X_train["ecpm"].values.reshape(-1, 1)
    X_test = X_test["ecpm"].values.reshape(-1, 1)
    lin_reg = RandomForestRegressor()
    lin_reg.fit(X_train, y_train)
    print(lin_reg.score(X_test, y_test))
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05)
    
    xgboost_model.fit(X_train, y_train)
    y_pred = xgboost_model.predict(X_test)
    # Evaluate the model using root mean squared error (RMSE)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse:.2f}")

    



# %%
train_and_test_model(df)
# %%
predictions
# %%
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(lin_reg, f)
# %%
df = df.drop([ 'device','error'], axis=1)
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
# lin_reg = RandomForestRegressor()
# lin_reg.fit(X_train, y_train)
# print(lin_reg.score(X_test, y_test))
# %%
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05)
# %%
xgboost_model.fit(X_train, y_train)

# %%

# %%
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators': [500, 750, 1000, 1100],
              'learning_rate': [0.4, 0.5, 0.6],
              'max_depth': [None]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=xgboost_model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters found
print(grid_search.best_params_)
# %%
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror',
                                  max_depth=None, 
                                  n_estimators=1000, 
                                  learning_rate=0.4)
xgboost_model.fit(X_train, y_train)
# %%
y_pred = xgboost_model.predict(X_test)
# Evaluate the model using root mean squared error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.2f}")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")
# %%
