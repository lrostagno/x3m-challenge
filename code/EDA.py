# %%
from utils import *
# %%
import pandas as pd
import matplotlib.pyplot as plt

def plot_piechart(df, column):
    counts = df[column].value_counts()
    plt.pie(counts.values, labels=counts.index.values, autopct='%1.1f%%')
    plt.title(f'Distribution of {column}')
    plt.show()