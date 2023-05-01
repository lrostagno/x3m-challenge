# %%
import requests
import pandas as pd
# %%
path = '../Challege-Data.tsv'
df = pd.read_csv(path, sep='\t')
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
