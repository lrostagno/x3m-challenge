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
# %%
def unnest_rows(df, column):
    # TODO: Add try catch
    df.waterfall_result = df.waterfall_result.apply(literal_eval)
    df = df.explode(column)
    df = df.join(pd.json_normalize(df[column])).drop(column, axis=1)
    return df

    


# %%
def process_data(df):
    df.waterfall_result = df.waterfall_result.apply(literal_eval)
    df = unnest_rows(df, "waterfall_result")
    return df
# %%
df = pd.read_csv('../Challege-Data.tsv', sep='\t')

# %%
df.dtypes["app_id"]

# %%
import time
# %%
tic = time.time()
df_time = unnest_rows(df, "waterfall_result")
print(time.time() - tic)
# %%
df_time
# %%
# %%
tic = time.time()
df["waterfall_result"] = [literal_eval(x) for x in df.waterfall_result]
print(time.time() - tic)
# %%
# %%
df
# %%
df.info()
# %%
df.describe()
# %%
df.describe().T
# %%
df.columns
# %%
import matplotlib.pyplot as plt
# %%
df.info()

# %%
df.columns
# %%

country_counts = df['adtype'].value_counts()

# Create a pie chart
plt.pie(country_counts.values, labels=country_counts.index.values, autopct='%1.1f%%')

# Add a title
plt.title('Distribution of countries')

# Show the plot
plt.show()


# %%
def show_piechart(df, column):
    counts = df[column].value_counts()

    # Create a pie chart
    plt.pie(counts.values, labels=counts.index.values, autopct='%1.1f%%')

    # Add a title
    plt.title(f'Distribution of {column}')

    # Show the plot
    plt.show()

# %%

# %%
df.columns
# %%
print("asfd")
# %%
columns = ['connection_type']
for column in columns:
    show_piechart(df_concat, column)

# %%

# %%
df["waterfall_result"] = df["waterfall_result"].apply(ast.literal_eval)

# %%
# create a new DataFrame with the dictionaries in "waterfall_result" column
df_waterfall = pd.DataFrame([x for sublist in df["waterfall_result"] for x in sublist])

# reset the index of the new DataFrame
df_waterfall = df_waterfall.reset_index(drop=True)

# %%
df_waterfall.describe(percentiles=[.25, .5, .75, .99]).T
# %%
df_waterfall.corr()
# %%
import matplotlib.pyplot as plt

# %%
plt.hist(df_waterfall['latency'], bins=5, range=(0, 5), color='blue')
plt.show()
# %%
df_waterfall = df_waterfall[df_waterfall["latency"] <= 7981.2]
df_waterfall = df_waterfall[df_waterfall["latency"] <= 1000]
df_waterfall = df_waterfall[df_waterfall["ecpm"] <= 79]
# %%
# %%
df_waterfall.describe(percentiles=[.25, .5, .75, .99]).T
# %%
len(df_waterfall)

# %%

# %%
def train_and_test_model(df):
    X = df.drop("latency", axis=1)
    #X = df_waterfall["ecpm"]
    y = df["latency"]
    scaler = StandardScaler()
    X[["ecpm"]] = scaler.fit_transform(X[["ecpm"]])
    X_train, X_test, y_train, y_test = train_test_split(
                                            X, 
                                            y, 
                                            test_size=0.01, 
                                            random_state=42)
    x = X_train["ecpm"].values.reshape(-1, 1)
    x_test = X_test["ecpm"].values.reshape(-1, 1)
    lin_reg = LinearRegression()
    lin_reg.fit(x, y_train)
    return lin_reg.score(x_test, y_test)
    



# %%

train_and_test_model(df_concat)
# %%
predictions
# %%
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(lin_reg, f)
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
df
# %%
# unnest the 'scores' column
df_unnested = pd.json_normalize(df['scores'].explode())
df_unnested
# %%
# join the unnested DataFrame with the original DataFrame
df_final = df.drop(columns='scores').join(df_unnested)

# %%
df_final
# %%
data

# %%
from ast import literal_eval
df
# %%
df
# %%
# combine columns to datetime the drop them


# %%
df_concat.loc[0]
# %%
df_concat.describe().T
# %%
df_concat = df_concat[df_concat["latency"] <= 2000]
df_concat = df_concat[df_concat["ecpm"] <= 79]
# %%
df_concat.describe()
# %%
df_concat.get_dummies

# %%
def 

country_pct = df['country'].value_counts(normalize=True) * 100
# %%
country_pct
# %%
# create a list of countries that have less than 5% frequency
low_freq_countries = country_pct[country_pct < 5].index.tolist()
# %%
low_freq_countries

# %%
# replace the names of these countries with the name of the new class
df_concat['country'] = df_concat['country'].replace(low_freq_countries, 'Rest of the world')
# %%
import pandas as pd

def replace_low_freq_values(df, col_name, freq_threshold):

    df_new = df.copy()
    
    col_pct = df_new[col_name].value_counts(normalize=True) * 100
    low_freq_values = col_pct[col_pct < freq_threshold].index.tolist()
    
    df_new[col_name] = df_new[col_name].replace(low_freq_values, 'Rest of the world')
    
    return df_new
# %%
df_concat = replace_low_freq_values(df, "connection_type", freq_threshold= 5)
# %%
one_hot = pd.get_dummies(df_concat['connection_type'])
# %%
one_hot
# %%
# %%
# Add the one-hot encoded columns to the original dataframe
df = pd.concat([df, one_hot], axis=1)
# %%
len(df_concat)
# %%
