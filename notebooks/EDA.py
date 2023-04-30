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
# %%
def unnest_rows(df, column, explode=False):
    df[column] = df[column].apply(literal_eval)
    # TODO: Add try catch
    if explode:
        df = df.explode(column)
    df.reset_index(inplace=True)
    df = pd.concat([df, pd.json_normalize(df[column])], axis=1)
    df = df.drop([column], axis=1)
    return df

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
    df = unnest_rows(df, "waterfall_result", explode=True)
    df = unnest_rows(df, "device")
    df["area"] = df["w"] 
    df = df.drop(["id", "event_id","level_0", "index", "event_time", "app_id", 
                  "user_id", "auction_id", "model", "hwv", "error", "partner", "w", "h"], axis=1)
    df = remove_outliers(df)
    df = replace_low_freq_values(df, "country", 8)
    df = replace_low_freq_values(df, "connection_type", 3)
    #df = replace_low_freq_values(df, "partner", 1)
    df = one_hot_encode(df, "adtype")
    df = one_hot_encode(df, "connection_type")
    df = one_hot_encode(df, "country")
    #df = one_hot_encode(df, "partner")
    df = encode_binary_feature(df, "platform")
    df = encode_binary_feature(df, "type")
    mean_ppi = df['ppi'].mean()
    df['ppi'].fillna(mean_ppi, inplace=True)
    scaler = StandardScaler()
    df[["ecpm", "area", "memory_total","ppi"]] = scaler.fit_transform(df[["ecpm", "area", "memory_total", "ppi"]])
    df = df.drop_duplicates()
    return df



# %%
df.corr()
# %%
df = clean_data()
# %%
pd.set_option('display.float_format', lambda x: '%.5f' % x)
df.corr()
# %%
df.describe(percentiles=[.01, .25, .5, .75, .99]).T
# %%
df.columns
# %%
df.head()
# %%


df_train = df.copy()
X = df_train[["platform", "type", "ppi", "area", "ban",	"itt", "rew", "AR",	"BR","MX","Other country","US"]]
y = df_train["latency"]
X_train, X_test, y_train, y_test = train_test_split(
                                        X, 
                                        y, 
                                        test_size=0.01, 
                                        random_state=42)
# X_train = X_train["ecpm"].values.reshape(-1, 1)
# X_test = X_test["ecpm"].values.reshape(-1, 1)

# %%
df.corr

# %%
df.columns
# %%
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', 
                                  n_estimators=1000, 
                                  learning_rate=0.1)
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
X
# %%
df.columns
# %%
df.corr()
# %%
df["type"].unique()
# %%
df.head()
# %%
1.05844 * 1.47935
# %%

df["area"] = df["w"] *df["h"]
# %%
df["area"]
# %%
