# %%
import requests
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
data_path = os.environ.get('DATA_PATH')
# %%
df = pd.read_csv(data_path, sep='\t')
json_data = df.to_json(orient='records')
import requests
api_endpoint = 'http://localhost:5000/predict'
headers = {'Content-Type': 'application/json'}
response = requests.post(api_endpoint, headers=headers, data=json_data)
if response.status_code == 200:
    print(response.json())
else:
    print('Error:', response.status_code)



# %%
import json
[d["latency"] for d in json.loads(response.json()["predictions"])]
# %%
df
# %%
l = 0
k = 99999999999999
df_l_to_k = df.iloc[l:k]
# %%
len(df_l_to_k) == len(df)
# %%
len(df_l_to_k)
# %%
len(df)
# %%
