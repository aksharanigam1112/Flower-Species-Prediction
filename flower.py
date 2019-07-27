import numpy as np
import pandas as pd

pd.set_option('display.width', 175)
pd.set_option('display.max_column', 8)
pd.set_option('precision', 2)

import matplotlib.pyplot as plt
import  seaborn as sbn
import  warnings
warnings.filterwarnings(action='ignore')

hnames = ['sepal length' ,'sepal width' ,'petal length' ,'petal width','class']
df = pd.read_csv('iris.csv',names = hnames)

print("\n\n",df.describe(include='all'))

print("\n Shape of the data set : ",df.shape)

print("\nCheck the count of null values for each feature :  \n", pd.isnull(df).sum())

arr = df.values
x = np.array(arr[:,0:4])
y = np.array(arr[:,4])

# print(x)

