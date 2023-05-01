# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from utils import evaluate_model, clean_data
import pickle
# %%
from dotenv import load_dotenv
load_dotenv()
data_path = os.environ.get('DATA_PATH')
df = clean_data(data_path)

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

model = RandomForestRegressor(n_estimators=1)
model.fit(X_train, y_train)
# %%
evaluate_model(model, X_test, y_test)

# %%
# Save the model as a pickle file
filename = 'model_API.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

# %%

params = {'objective': 'reg:squarederror',
          'colsample_bytree': 0.3, 
          'learning_rate': 0.05,
          'max_depth': None, 
          'alpha': 10,
          'n_estimators': 30} 


model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)
# %%
evaluate_model(model, X_test, y_test)


# %%

model = xgb.XGBRegressor(objective ='reg:squarederror')

# Set up the grid search
params = {
    'max_depth': [None],
    'learning_rate': [0.05, 0,1],
    'n_estimators': [500, 1000]
}
grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Save the best model as a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

# Print the best parameters and score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

