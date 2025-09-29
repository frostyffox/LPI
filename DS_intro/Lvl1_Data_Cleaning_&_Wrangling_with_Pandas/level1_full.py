# DATA CLEANING AND WRANGLING WITH PANDAS

# DATA:
# - PLANETARY HEALTH SUBSET, with environmental and health indicators
# - STUDENT HEALTH SUBSET, messy servedy from data students

import pandas as pd
import matplotlib.pyplot as plt

# loading ds
planetary = pd.read_csv("planetary_health_subset.csv")
student = pd.read_csv("student_health_subset.csv")

# getting the first lines of the two ds; 
# if no paramn
planetary.head(), student.head()

# structure inspection
planetary.info() # info on df
planetary.isna().sum()

planetary['date'] = pd.to_datetime(planetary[['year','month']].assign(day=1))
planetary.set_index('date', inplace = True)

# DEALING WITH MISSING VALUES
# identify if any columns have missing values

# dfname.columns[df.isna().any()]
# isna(): ; any():
missing = planetary.columns[planetary.isna().any()]

# drop, fill or interpolate?
# DROPPING: used when a columns has too many missing values
# also, if a row is missing values in critical columns

# FILLING: filling categorical data with the 'mode'
# or numerical (relatively stable) values with mean or median

# INTERPOLATING: to be used on time series data
# when data change smoothly
# to estimate missing points between the known ones
# in this case interpolating is the best choice!

planetary.interpolate(inplace=True) #-> linear interpolation
# .interpolate estimates missing values from nearby values

#planetary.interpolate(method='time', inplace=True)

# CHECKING FOR AND REMOVING OUTLIERS
# outliers: PM2.5 < 0 or LST > 60
pm25_out = planetary[planetary['pm25']<0]
lst_out = planetary[planetary['land_surface_temp'] > 60]

#removing
planetary = planetary[(planetary['pm25']>=0) & (planetary['land_surface_temp']<=60)]
#check
print("Remaining rows:", len(planetary))

# ANALYSIS
# average PM2.5 per country, per year

#1. make sure i have the right columns
#2. use groupby
avg_pm25 = planetary.groupby(['county','year'])['pm25'].mean().reset_index()
# groupby(['country','year']) -> splitting the data by country and year
# ['PM2.5'].mean() -> calculate the mean FOR EACH GROUP!
print(" \n*** Average PM2.5 per County and Year *** ")
print(avg_pm25.head())

# PLOTTING
# asthma prevalence vs PM2.5
# x = predictor = independent variable -> PM2.5
# y = predicted = dependent variable -> asthma
x = planetary['pm25']
y = planetary['asthma_prev']
plt.scatter(x,y)
plt.xlabel('pm25')
plt.ylabel('Asthma prevalence (%)')
plt.title('Asthma vs PM2.5')
plt.show()

#_________WORK ON STUDENT DATASET__________

# 1) Clean `gender` column into standardized values [Male, Female, Other].
def standard_gender(g):
    if str(g).strip().lower() in ['m','male']:
        return 'Male'
    elif str(g).strip().lower() in ['f','female']:
        return 'Female'
    else:
        return 'Other'

student['gender'] = student['gender'].apply(standard_gender)

# ALTERNATIVE:
# student['gender_clean'] = student['gender'].str.lower().map({
#    'm': 'Male', 'male': 'Male',
#   'f': 'Female', 'female': 'Female',
#   'other': 'Other'
#})
print("\n *** Student gender format fixed *** ")
print(student[['gender']].head())

    
# 2) Convert `stress_level` to ordinal scale: Low=1, Medium=2, High=3.
student['stress_clean'] = student['stress_level'].str.lower().map({
    'low' : 1, 'medium': 2, 'high': 3
})
print("\n *** Stress level expression normalised ***")
print(student[['stress_level', 'stress_clean']].head())

# 3) Fix `gpa` column (convert commas, replace 'four' with 4.0, coerce to numeric).
# 3.a) numbers with commas : n,m -> n.m
student['gpa_ok'] = student['gpa'].str.replace(',','.', regex=False)

# 3.b) 'four' -> 4.0
student['gpa_ok'] = student['gpa_ok'].str.lower().replace({'four':'4.0'})

# 3.c) everything -> numeric (except: NaN)
student['gpa_ok'] = pd.to_numeric(student['gpa_ok'], errors='coerce')

#check the result
print("\n *** Student GPA data format cleaned *** ")
print(student[['gpa','gpa_ok']].head())

# Handle missing values in `sleep_hours` (impute with median).
# PRE_WORK: convert to numeric
student['sleep_hours'] = pd.to_numeric(student['sleep_hours'], errors='coerce')

# A) compute the median over the sleep_hours values
median_sleep = student['sleep_hours'].median()

# B) replace the missing values with median
student['sleep_hours'].fillna(median_sleep, inplace=True)

# check
print(student['sleep_hours'].head())


# 4) Detect and filter out impossible ages (<15 or >60).
age_impo = student[(student['age']<15) | (student['age']>60)]

student = student[(student['age']>=15) & (student['age']<=60)]

# Group by major: average GPA and stress.
#computing average GPA and stress for each major
gsm = student.groupby('major').agg ({ # gsm = gpa-stress_major
    'gpa_ok':'mean',
    'stress_clean':'mean'
    }).reset_index()

# Explore relationships: Does more coffee correlate with less sleep?
# x = independent variable = coffee cups
# y = dependent variable = sleep hours

x = student['coffee_cups']
y = student['sleep_hours']
plt.scatter(x,y)
plt.xlabel('Coffee cups per day')
plt.ylabel('Hours of sleep')
plt.title('Coffee consumption vs Sleep time')
plt.show()

#compute a correlation coefficient
correlation_c = student['coffee_cups'].corr(student['sleep_hours'])
if correlation_c == -1:
    print("The more coffe you drink, the less you will sleep!")
elif correlation_c < -0.5:
    print("Coffee affects your sleep alot!")
elif correlation_c == 1:
    print("Drink as much coffee as you want: it won't affect your sleep")
elif correlation_c > 0.5:
    print("Coffee does not really affect your sleep")
else:
    print("No relationship")