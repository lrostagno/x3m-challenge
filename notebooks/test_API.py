# %%
import requests
import json
import pandas as pd
from EDA import clean_data
# %%
url = 'http://localhost:5000/predict'
data = {'a': 4, 'b': 56}
# %%
path = '../Challege-Data.tsv'
df = pd.read_csv(path, sep='\t')

df.head()
# %%
response = requests.post(url, json=data)
if response.status_code == 200:
    print(response.json())
else:
    print('Error:', response.status_code)
# %%
response.json()
# %%
import pickle
import xgboost as xgb

# Load the model from file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
# %%
model.feature_names_in_
# %%
a = df.head(2).to_json()
# %%
type(a)
# %%
b = json.loads(a)
# %%
b.keys()
# %%
b["event_id"]
# %%
data = df.head(5).to_dict(orient='records')
# %%
len(data)
# %%
# %%
# %%
json_data = json.dumps(data)

# %%

# %%
I have an array like this:
['partner_APS', 'partner_AdColony', 'partner_AdMob', 'partner_AdView', 'country_AR', 'country_BR', 'country_US']
And i need to get a dictionary like this:
{'partner': ['APS', 'AdColony', 'AdMob', 'AdView'], 'country' : ['AR', 'BR', 'US']}
# %%

def split_array_to_dict(arr):
    d = {}
    for elem in arr:
        key, value = elem.split('_', 1)
        if key in d:
            d[key].append(value)
        else:
            d[key] = [value]
    return d
# %%
arr = ['partner_APS', 'partner_AdColony', 'partner_AdMob', 'partner_AdView', 'country_MX','country_AR', 'country_BR', 'country_US']
# %%
my_dict = split_array_to_dict(arr)
# %%
def split_array_to_dict(arr):
    d = {}
    [d.setdefault(k, []).append(v) for k, v in [elem.split('_', 1) for elem in arr]]
    return d

# %%



# %%
arr = split_array_to_dict(arr)
# %%
arr.keys()
# %%
for i in range(len(df_test)):
    column = "country"
    print(df_test.iloc[i][column] in my_dict[column])
    if df_test.iloc[i][column] in my_dict[column]:
        df_test.iloc[i][f"column_{my_dict[column]}"] = 1
    else:
        df_test.iloc[i][f"other_{column}"] = 1

    if i == 5:
        break
# %%
df.iloc[1]
# %%
df_test = df.copy()
# %%

# %%
df_test["BR"].value_counts()
# %%

# ESTA LISTA YA LA TENGO
new_columns = ["country_AR", "country_BR", "country_MX", "country_US", "other_country"]
df_test = pd.concat([df_test, pd.DataFrame(0, index=df_test.index, columns=new_columns)], axis=1)
# %%
df
# %%

df_test

# %%
