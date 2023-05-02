# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from utils import evaluate_model, download_and_clean_data
import pickle
from dotenv import load_dotenv
load_dotenv()
import os
data_path = os.environ.get('DATA_PATH')
df = download_and_clean_data(data_path)

# %%
df_train = df.copy()
cols = df_train.columns.tolist()
X = df_train[cols].drop(["latency"], axis=1)
y = df_train["latency"]
X_train, X_test, y_train, y_test = train_test_split(
                                        X, 
                                        y, 
                                        test_size=0.01, 
                                        random_state=42)
# %%
model = xgb.XGBRegressor(objective ='reg:squarederror')
params = {
    'max_depth': [None],
    'learning_rate': [0.05, 0,4],
    'n_estimators': [500, 1000]
}
grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
with open('final_model.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
# %%
X_train.info()
# %%
len(X_train)
# %%
# Training final model with all the data
# %%
model = xgb.XGBRegressor(objective ='reg:squarederror',
                         n_estimators=1000,
                         learning_rate=0.4,
                         max_depth=None)
# %%
model.fit(X, y)
# %%
evaluate_model(model, X, y)
# %%
with open('latency_xgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)
# %%
