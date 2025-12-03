import numpy as np

# creating a ds
ds = np.array([[[1,2,3,4],[6,7,8,9],[10,11,12,13]]])

type(ds)

ds.size

ds.shape
ds.dtype #datatype

ds.transpose()

#((rows,cols),dtype) -> make an empty array
np.empty((4,4), dtype=int)

np.ones(6) #empty arr of 6 cols
np.zeros((4,8), dtype= float)

import pandas as pd
df = pd.read...
df.head()

#check for null values:
df.isnull()
df.isnull().sum()
df.isnull().sum().sum()

#method 1: drop
df.shape()
df2 =df.dropna()
df2.shape()

df3 =df.dropna(axis=1)
df3.shape()

#df.dropna(how= any / all)
#df.dropna(inplace=True) -> re[place the nulls 

df.fillna(method= 'ffill', axis=1) 
#forward fill: fills the null with the previous value
#axis: default = 0
# axis = 1-> takes the previous column

#more precise
df['Col_name'].fillna(value= df['Col_name'].mean())

#bfill: fill with the value right after

#fillna(inplace); this paramenter replace the 