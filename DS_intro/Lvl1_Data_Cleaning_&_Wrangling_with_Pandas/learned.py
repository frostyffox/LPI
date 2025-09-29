#_______PANDAS_______
import pandas as pd
# Reading csv

# file_name as fn
fn = pd.read_csv("file_path.csv")

# getting the first element of the file
fn.head()

#inspect the structure and getting info
fn.info()
fn.isna().sum() 
# .isna() replaces all the df NAN values with True, otherwise False
# useful to track the NaN values

# finding missing elements
missing = fn.columns[fn.isna().any()]

#______dealing with missing values_______

#1: when a column misses too many values or a row is missing crucial col values
# use DROPPING

#2: FILLING
# use 'mode' to fill in categorical data
# use 'mean' or 'median' for numeric values

#3: INTERPOLATING
# when you have some gaps that can be filled with near values
fn.interpolate(inplace=True)

#_____dealing with outliers_______
#ex: i want the values in col 'col1' to be > 3; 
# i want the values in 'col2'  to be ==5
col1_out = fn[fn['col1']>= 3]
col2_out = fn[fn['col2'] != 5]

#removing values
fn = fn[(fn['col1']<3) & (fn['col2'] == 5)]

#printing the remaining data
print("Remaining rows:", len(fn))

# calculating the average 'col1' per 'category1', 'category2'
avg_col1 = fn.groupby(['cat1','cat2'])['col1'].mean().reset_index()

#______Replacing/eliminating values_______

#If column col3 has too many messy values, 'map' is not appropriate


#______GROUP BY________
# "Groupin by 'col'" means grouping the elements of the table by their value for the column 'col'

# .agg() -> computes the average for eache element of a dictionary

# ex: 
avg_by_somehting = fn.groupby('col3').agg({
    'col1':'mean',
    'col5':'mean'
}).reset_index()