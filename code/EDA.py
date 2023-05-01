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

def unnest_rows(df, column, explode=False):
    df[column] = df[column].apply(literal_eval)
    # TODO: Add try catch
    if explode:
        df = df.explode(column)
    df.reset_index(inplace=True)
    #df = pd.concat([df, pd.json_normalize(df[column], meta_prefix=column, sep='_')], axis=1)
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

def plot_piechart(df, column):
    counts = df[column].value_counts()
    plt.pie(counts.values, labels=counts.index.values, autopct='%1.1f%%')
    plt.title(f'Distribution of {column}')
    plt.show()
    
def remove_outliers(df):
    # TODO: Parametrizar esta función
    df = df[df["latency"] <= 1000]
    # df = df[df["ecpm"] <= 2]
    return df

def one_hot_encode(df, column):
    one_hot = pd.get_dummies(df[column], prefix=column, prefix_sep=",")
    df = pd.concat([df, one_hot], axis=1)
    df = df.drop(column, axis=1)
    return df


def clean_data(path, training=False):
    df = pd.read_csv(path, sep='\t')
    df = unnest_rows(df, "waterfall_result", explode=True)
    df = unnest_rows(df, "device")
    df["area"] = df["w"] * df["h"]
    df = df.drop(["id", "event_id","level_0", "index", "event_time",
                  "user_id", "auction_id", "model", "hwv", "error", "w", "h", 
                  "memory_total"], axis=1)
    df = remove_outliers(df)
    df = replace_low_freq_values(df, "country", 8)
    df = replace_low_freq_values(df, "connection_type", 3)
    df = replace_low_freq_values(df, "partner", 1)
    df = replace_low_freq_values(df, "app_id", 5)
    df = one_hot_encode(df, "adtype")
    df = one_hot_encode(df, "connection_type")
    df = one_hot_encode(df, "country")
    df = one_hot_encode(df, "partner")
    df = one_hot_encode(df, "app_id")
    df = one_hot_encode(df, "platform")
    df = one_hot_encode(df, "type")
    # TODO: ver esto de los nan
    mean_ppi = df['ppi'].mean()
    df['ppi'].fillna(mean_ppi, inplace=True)
    #scaler = StandardScaler()
    #df[["ecpm", "area", "ppi"]] = scaler.fit_transform(df[["ecpm", "area", "ppi"]])
    df = df.drop_duplicates()
    return df

# %%

path = '../Challege-Data.tsv'
df = clean_data(path)
# %%

# %%
df_train = df.copy()
cols = df_train.columns.tolist()
# X = df_train.drop(["latency", 'AdMob', 'AdView', 'AppLovin',
#        'Chartboost', 'DFP', 'Fyber', 'HyprMX', 'InMobi', 'IronSource',
#        'Mintegral', 'Ogury', 'Other partner', 'Pangle', 'Startio', 'Unity',
#        'Vungle'], axis=1)
#X = df_train.drop(["latency"], axis=1)
X = df_train[cols].drop(["latency"], axis=1)
y = df_train["latency"]
X_train, X_test, y_train, y_test = train_test_split(
                                        X, 
                                        y, 
                                        test_size=0.01, 
                                        random_state=42)
# %%
X_train
# %%

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=10)

model.fit(X_train, y_train)
# %%
def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse:.2f}")

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.5f}")
    print(f"Mean Absolute Error: {mae:.5f}")
    print(f"R-squared: {r2:.5f}")

evaluate_model(model, X_test, y_test)
# %%
evaluate_model(model, X_train, y_train)

# %%

# %%
X_train
# %%
X_test
# %%
import pickle


# Save the model as a pickle file
filename = 'model_API.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
# %%
X_test
# %%
y_test
# %%
model.score(X_test, y_test)
# %%
y_train.describe()
# %%
y_test.describe()
# %%
import xgboost as xgb
params = {'objective': 'reg:squarederror', # use squared error as the objective function
          'colsample_bytree': 0.3, # fraction of columns to use when constructing each tree
          'learning_rate': 0.05, # step size shrinkage used to prevent overfitting
          'max_depth': None, # maximum depth of each tree
          'alpha': 10, # L1 regularization term on weights
          'n_estimators': 30} # number of trees in the model

# Create XGBoost regressor object
model = xgb.XGBRegressor(**params)
# %%
model.fit(X_train, y_train)
# %%
evaluate_model(model, X_test, y_test)
# %%
evaluate_model(model, X_train, y_train)
# %%
y_test.describe()
# %%
from sklearn.model_selection import GridSearchCV
model = xgb.XGBRegressor(objective ='reg:squarederror')

# Set up the grid search
params = {
    'max_depth': [None],
    'learning_rate': [0.05],
    'n_estimators': [500, 1000, 2000, 3000]
}
grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Save the best model as a pickle file
with open('best_model.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
# %%
df_test = pd.DataFrame({"A": ["asdf", "asdf"], "B": ["asdf", "324"]})
# %%
df_test[["A", "B"]].to_json(orient="records")
# %%