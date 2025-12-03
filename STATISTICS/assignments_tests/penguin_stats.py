
"""penguin_final_dirocco.ipynb

Original file is located at
    https://colab.research.google.com/drive/1m_D5PQhEZ9Iyqjzr77NkB9D9JbVYL_D8

# Statistic Group Assignment 1 - Penguins

**Goals**
- Practice summary statistics and data visualization.
- Explore data transformations and missing data handling.
- Apply parameter estimation and compute confidence intervals.
- Formulate insights based on data analysis.
Homework link: https://mahendra-mariadassou.pages-forge.inrae.fr/m1-aire-stats-course/Homework/HW1_Penguins/aire_m1_homework1.html
"""

# Imports
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""# 1 Load Dataset"""

data = sns.load_dataset("penguins")

data.head()

"""# 2 Data Summary and Basic Exploration
Describe the dataset, calculate summary statistics

Question 1: What are the types of each variable in this dataset? Identify whether each column is numerical or categorical.

Question 2: Calculate and report in a table the mean, median, and standard deviation for each of the numerical columns. Comment on any patterns or differences you observe.

Question 3: Repeat the operations but within each species. Comment the differences between the results obtained on the full dataset and those obtained per species.
"""

# 2.1 datatypes
data.info()

"""
Species, island, sex are categorical data; others are numerical figures of penguins.
"""

# 2.2 -> Calculate stats data of the whole dataset
data.describe(include='all')

# count only numeric data

numeric_data = data[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
result_whole = numeric_data.agg(['count','mean','median','std']).reset_index()
result_whole = result_whole.rename(columns={'index': 'stats summary'})
result_whole

"""
Bill length
- The mean is slightly smaller than the median -> slightly **left skewed**

Bill depth
- The **mean and median are close**

Flipper length
- The mean is greater than the median -> slightly **right skewed**

Body mass
- The mean is greater than the median, and the std dev is very wide. However, **the order of magnitude of the standard deviation is "coherent"** with the one of the body mass measurements.

More about the order of magnitude:
- Bill length and bill depth: mean, median ~ 10^1 * n; std dev ~ 10^1 * n
- Flipper length: mean, median ~ 10^2 * n; std dev ~ 10^1 * n -> gap of 1 order of magnitude
- Body mass: mean , median ~ 10^3 * n and std dev ~ 10^2 * n (but closer to 1000 than to 100) -> gap of 1 order of magnitude

penguin pic: https://cimioutdoored.org/wp-content/uploads/2020/03/penguin-swimming-300x169.jpg
"""

# 2.3 -> Calculate stats data with categories
numeric_data_species = data[['species', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
result_species = numeric_data_species.groupby('species').agg(['count','mean','median','std']).reset_index()
result_species

"""
Comment the differences between the results obtained on the full dataset and those obtained per species.

- In general the summary statistics of the full dataset do not tell us much about differences between different species.
- Once we group the data by species, the standard deviations of all numeric variables become smaller. This happens because each species has some consistant physical traits.
    - For bill length, the mean and median are around 44mm for the whole population. However we can see that Chinstrap and Gentoo penguins have longer bills and Adelie penguins have shorter ones.
    - For bill depth, Gentoo penguins have shallower bills, with both a low standard deviation and a low mean in comparison to those of the whole population
    - For body mass and flipper length, Gentoo penguins are generally larger and heavier than the other species, which explains why the mean and median of these variables in the full dataset are noticeably higher than those of Adelie and Chinstrap penguins.

# 3 Missing data
Question: Are there any missing values in the dataset? If so, how would you handle them (the answer can depend on the variable) ? Write down your approach and give a few lines of rationale.

Action: If any missing values were found, apply the method you chose to handle them (e.g., remove rows, impute values, etc.). In the rest of the homework, work only on the cleaned version of the dataset.
"""

# Checking missing data
data.isnull().any()

# Returns true columns have missing values

data.isnull().sum()

# Select columns with missing values
cols_missing_data = data.columns[data.isnull().any()]

# Select only rows that have at least one missing value in these columns
missing_rows_df = data[data[cols_missing_data].isnull().any(axis=1)]

# Display the result
missing_rows_df

"""
There are 11 rows containing missing data.
- We first drop row 3 and row 339 directly, because they miss all the data, and they are only 2/344, we consider this is the case of 'Missing Completely at Random'.
- For the other 9 rows, the sex variable is missing. Originally we grouped penguins by species and sex, and drew boxplots about that (We will show it later). Indeed there are some statistical differences between females and males penguins within species, yet these differences are not distinct enough to infer sex based on the available measurements.
- Given that only 11 rows in total contain missing values, which is a very small portion of the dataset, we decided to drop them altogether to simplify the dataset and keep the analysis cleaner.
"""

# Removal of missing data
data = data.dropna().reset_index(drop=True)
data.info()

"""Now we are going to check the quantity of females and males for each species and the missing data for each sex and species for our coming analysis."""

#NEW
sex_count = data.groupby(["species","sex"]).size().unstack(fill_value=0)
sex_count

"""# 4 Visualizing Data
Create histograms, KDEs, boxplots for numerical variables
- Plot the distribution of each numerical variable using histograms or KDE plots.
    - Question: What do these distributions tell you about the spread and central tendency of each variable?
- Create boxplots to explore potential outliers in each variable.
    - Question: Are there any noticeable outliers? If so, describe their potential impact on the analysis. How many outliers are identified for each numerical variable by the IQR rule ?

## 4.1 Visualizing distributions of numerical variables
"""

# plot histograms
num = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
for col in num :
 plt.figure(figsize=(8,4))
 sns.histplot(data=data, x=col, kde=True, bins =30)
 plt.title(f"Distribution of {col}")
 plt.show()

"""
- Most variables have **moderate spread**.
- For most numerical variables **the mean and the median are close**, so the distributions are roughly symmetrical; even so, **body mass is slightly skewed to right** and **bill length is slightly skewed to left**.
- Bill length and flipper length reveal some **bimodality**; we saw this depends on the existence of subpopulation, that correspond to different species.
- In general, none of the distributions appear to be strictly condensed around the central value; we can explain this with the fact that each species “covers” different values of the different measurements.
- Chinstrap are the less represented ones, Adelie are the most numerous; but Chinstrap values tend to be in between the other 2 species, so we can map their contribution to the distribution frequency as the “joint” between Adelie and Gentoo."""

# plot flipper length distribution by species
sns.histplot(data=data, x='flipper_length_mm', hue='species',kde=True, bins=30)
plt.title("Flipper length distribution by species")
plt.show()

"""**Comments:**

Now we can clearly see that the distribution we plotted first did not look exaclty unimodal because they contained different sub-populations, corresponding to the different species.
"""

sns.histplot(data=data, x='bill_length_mm', hue='species',kde=True, bins=30)
plt.title("Bill length distribution by species")
plt.show()

"""**Comments:**

Same here!! Happy :)

## 4.2 Checking outliers
- Create boxplots to explore potential outliers in each variable.
    - Question: Are there any noticeable outliers? If so, describe their potential impact on the analysis. How many outliers are identified for each numerical variable by the IQR rule ?
"""

# Outliers with IQR
outlier_summ = {}

for col in num:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    l = Q1 - 1.5 * IQR
    u = Q3 + 1.5 * IQR

    outliers = data[(data[col] < l) | (data[col] > u)]
    outlier_summ[col] = {'count': len(outliers), 'min_fence': l, 'max_fence': u}

    print(f"{col}: {len(outliers)} outliers (IQR fence: {l:.2f}–{u:.2f})")

"""**Comments:**

Applying the IQR rule to the whole penguin population do not indicate any outlier. With the standard 1.5 × IQR threshold, the range is relatively wide, so no observations fall outside the threshold.
"""

# Z-score method

for col in num:
   z_scores = np.abs(stats.zscore(data[col].dropna()))
   outliers = np.sum(z_scores > 3)  # 3 standard deviations
   print(f"{col}: {outliers} potential outliers (z>3)")

"""**Comments:**

Even with the Z score there are no outliers detected. Happy.
- the dataset is "well behaved" with no extreme measurements.
- there is some variability, within the biological ranges, but it is not counting as "statistical" outlier.

**But, what if we group the data by species and sex?**

**Check below!**
"""

#Double checking the existence of ouliers with box plots

sns.set(style='whitegrid')

for col in num:
 plt.figure(figsize=(8,5))
 sns.boxplot(
     data=data, x='species', y=col, hue ='sex', palette = 'Set2'
 )

 plt.title(f"{col.replace('_', ' ').title()} by Species and Sex")
 plt.xlabel("Species")
 plt.ylabel(col.replace('_', ' ').title())
 plt.legend(title='Sex', loc='best')
 plt.tight_layout()
 plt.show()

"""**Comments:**

With the boxplots we can see the spreading of the values better and in fact, there is some variability. It probably accounts for very few individuals and their measurements are still within "biological variability".

Consistently with what we saw before, in each species, male tend to be bigger and heavier. There is some variability within each species and each sex.
- Gentoo penguins show the largest body mass and flipper length.
- Adelie are smallest and more "compact".

In fact, by grouping by species and sex, we are able to find some outliers:
"""

# If we do by gender and species, it shows outliers:

# Empty list to collect results
outlier_records = []

# Loop through each species–sex group
for (species, sex), group in data.groupby(['species', 'sex']):
    for col in num:
        # Drop NaN values for this column
        vals = group[col].dropna()
        if vals.empty:
            continue

        # Compute quartiles and IQR
        Q1 = vals.quantile(0.25)
        Q3 = vals.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = group[(group[col] < lower) | (group[col] > upper)]

        # Add each outlier as a record
        for idx, row in outliers.iterrows():
            outlier_records.append({
                'species': species,
                'sex': sex,
                'variable': col,
                'value': row[col],
                'lower_fence': lower,
                'upper_fence': upper,
                'row_index': idx
            })

# Convert to DataFrame
outlier_df = pd.DataFrame(outlier_records).reset_index()
outlier_df

"""Now we want to visualise the distribution for each variable, by species and sex. The plots include mean and quantiles."""

#Violin plots for outliers in each (numeric) variable, by sex and species

fig, axes = plt.subplots(1, len(num), figsize=(6 * len(num), 6), sharey=False)

# If only one column, axes is not an array
if len(num) == 1:
    axes = [axes]

for ax, col in zip(axes, num):
    sns.violinplot(
        data=data,
        x='species',
        y=col,
        hue='sex',
        split=True,
        inner='quartile',    #to visualise the quartiles
        palette='Set2',
        ax=ax
    )

    ax.set_title(f'{col.replace("_", " ").title()} by Species and Sex')
    ax.set_xlabel('Species')
    ax.set_ylabel(col.replace('_', ' ').title())

# Move legend outside
handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, title='Sex', loc='upper center',
           ncol=2, bbox_to_anchor=(0.5, 1.07))

plt.tight_layout()
plt.show()

"""**Comments:**

**Working with numerical variables and their outliers: we have normal distributions**

There are 24 outliers in total. Number of outliers for each variable:
- Bill length: 16
- flipper length: 4
- body mass: 2
- bill depth: 2

These outliers are found after grouping penguins by species and sex, more in details:
- Adelie:
  - Female:
    - Bill depth: 1 -> very close to upper fence, little effect (slight left skewed)
    - flipper length: 3 -> no significan effect (all very close to the fences)

  - Male:
    - bill length: 5 -> 3 > of the upper fence, 2 < of the lower fence; slight skew on the left
    - bill depth: 1 -> very light skewed on left
    - flipper length: 1 -> approximately no effect

****

- Chinstrap:
  - Female:
    - Bill length: 5 -> approximately no effect
    - body mass: 1 -> just 1 outlier causes a slight right skew

  - Male:
    - bill length: 1 -> approximately no effect
    - body mass: 1 -> slight right-skewed

****

- Gentoo:
  - Female:0
  Even so, all the variable distributions have light skew (see plots below)

  - Male:
    - bill length: 5; 4 out of 5 are slightly bigger than upper bound; very slight left skew

To sum up the *impact of the outliers on computations* :

The outliers are quite close to either the Lower or Upper fence.

# 5. Exploring Relationships between Variables
Scatter plots and discussions on potential correlations
- Use adapted plots to investigate relationships between pairs of features. You should examine at least (but can consider also consider others):
    - bill length vs. bill depth
    - body mass vs. flipper length
    - body mass vs. species
- Question: Do any variables seem to have a linear or other type of relationship? Describe any trends or patterns you observe between numerical variables using the appropriate measure. What do you conclude ?

## 5.1 Bill length vs bill depth
"""

sns.pairplot(data, hue="species")

# bill length vs. bill depth (by species)

sns.lmplot(
    data=data,
    x='bill_length_mm',
    y='bill_depth_mm',
    hue='species',
    height=6,
    aspect=1.2,
    scatter_kws={'s': 70, 'alpha': 0.7, 'edgecolor': 'black'},
    legend=False   # turn off the automatic legend
)

plt.legend(title='Species', loc='lower right', fontsize=9, title_fontsize=10)
plt.title('Bill Length vs Bill Depth by Species')
plt.xlabel('Bill Length (mm)')
plt.ylabel('Bill Depth (mm)')
plt.tight_layout()
plt.show()

"""In all species we see a positive correlation between bill length and bill depth. the "degree of correlation" is similar for the three species (at least from the plots... but let's calculate...)"""

# Calculate correlation by species:
correlation_by_species = data.groupby('species').apply(
    lambda df: df['bill_length_mm'].corr(df['bill_depth_mm'])
)

print(correlation_by_species)

"""**Comments:**

In the pairplot, the relationship appears strong at first glance, but when examined more closely, the correlation is not as strong as it seems. And we can calculate the correlation coefficient as shown above.

We can conclude that:
- Among Adelie penguins, bill length and depth are **weakly related**.
- Among Chinstrap and Gentoo penguins respectively, there is a **moderate positive** linear relationship. As bill length increases, bill depth tends to increase.

## 5.2 body mass vs. flipper length
"""

# FacetGrid: one subplot per species
g = sns.FacetGrid(
    data,
    col='species',
    height=5,
    aspect=1
)

# Draw scatter colored by sex, regression for species
def scatter_with_regression(data, **kwargs):
    # Draw all points colored by sex
    sns.scatterplot(
        data=data,
        x='flipper_length_mm',
        y='body_mass_g',
        hue='sex',
        s=70,
        alpha=0.6,
        edgecolor='black',
        **kwargs
    )
    # Draw single regression line for all points in this facet
    sns.regplot(
        data=data,
        x='flipper_length_mm',
        y='body_mass_g',
        scatter=False,
        line_kws={'lw':2, 'color':'black'}
    )

g.map_dataframe(scatter_with_regression)

# Add a legend for sex
g.add_legend(title='Sex')

# Axis labels and titles
g.set_axis_labels('Flipper Length (mm)', 'Body Mass (g)')
g.set_titles(col_template="{col_name} Species")
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Body Mass vs Flipper Length by Species', fontsize=16)

plt.show()

# Calculate correlation by species:
correlation_by_species = data.groupby('species').apply(
    lambda df: df['flipper_length_mm'].corr(df['body_mass_g'])
)

print(correlation_by_species)

"""**Comments:**

A similar regression analysis is conducted for body mass and flipper length.

- Adelie: body mass and flipper length are **weakly related**.
- Chinstrap and Gentoo: there is a **moderate positive** linear relationship.
- Interestingly, penguins of different sexes form distinct clusters in the plots **especially for Adelie and Gentoo**, suggesting that male and female penguins may exhibit characteristic differences in body mass and flipper length, which is coherent with expected biological patterns.

## 5.3 body mass vs. species
"""

# we already have some boxplots above and i wanna try violinplot
sns.violinplot(x='species', y='body_mass_g', data=data)

sns.violinplot(x='species', y='body_mass_g', data=data, inner=None)
sns.swarmplot(x='species', y='body_mass_g', data=data, color='k', alpha=0.6)

"""**Comments:**

For species and body mass, we choose violinplot to examine the relationship between quantitative and qualitative data.

- The body mass of Adelie and Chinstrap is approximately a **normal distribution**, but Chinstrap showcases **a stronger central tendancy**.
- Gentoo penguins exhibit **a bimodal tendency**, which may be attributed to differences in sex (see the boxplots in Part 4), as male and female Gentoo penguins differ more in body masses in comparison to Adelie and Chinstrap.

## 6. Random Sampling and Confidence Intervals
- Take a random sample of n = 30 observations from the dataset (you can use `random.sample()` from `sklearn` or a similar function).
    - Question: Calculate the mean of bill length for this sample. How does it compare to the mean of the entire dataset?
- Generate a 95% confidence interval for the mean bill length of this sample using the formula:
    - sample mean +- 1.96*standard deviation / sqaure root of n (sample mean +- standard error)
- Does your confidence interval contains the population mean of bill length ?
- Repeat the operation 100 times and report the number of times your confidence interval contains the population mean. Comment on the result.
"""

# get a sample of 30 and sample mean
sample_30 = data.sample(n=30, replace=True, random_state=42)
mean_bill_length = sample_30['bill_length_mm'].mean()
print(mean_bill_length)

# result is 44.50666666666667
# population mean: 43.921930

"""The sample mean is higher than the population mean."""

import math

# sample_30 = data.sample(n=30, replace=True, random_state=42)
mean_bill_length = sample_30['bill_length_mm'].mean()
std_bill_length = sample_30['bill_length_mm'].std()

SE = std_bill_length / math.sqrt(30) # standard error
margin = 1.96 * SE # margin of error

CI_lower = mean_bill_length - margin
CI_upper = mean_bill_length + margin

print(f"95% CI: ({CI_lower:.2f}, {CI_upper:.2f})")
# result: 95% CI: (42.31, 46.70)

"""**Comments**:
The sample mean of bill length is 44.506 , falling in the 95% Confidence Interval zone.
"""

# repeat one hundred time and check if population mean falls into confidence intervals

population_mean = 43.921930
n_simulations = 100
sample_size = 30
count_in_CI = 0

for i in range(n_simulations):
    # Take a random sample of 30 with replacement
    sample_30 = data.sample(n=sample_size, replace=True)

    # Sample mean and std
    mean_bill_length = sample_30['bill_length_mm'].mean()
    std_bill_length = sample_30['bill_length_mm'].std()

    # Standard error and margin
    SE = std_bill_length / math.sqrt(sample_size)
    margin = 1.96 * SE

    # Confidence interval
    CI_lower = mean_bill_length - margin
    CI_upper = mean_bill_length + margin

    # Check if population mean falls inside CI
    if CI_lower <= population_mean <= CI_upper:
        count_in_CI += 1

print(f"The population mean fell inside the 95% CI {count_in_CI} times out of {n_simulations}")

"""**Comments**:

We did not choose a fixed randomness here and we can try computing many times.

The percentage of the sample mean falling inside the confidence interval varies each time, but in general it is not very far from 95%.

This result makes sense. According to the definition of confidence interval, if we repeatedly take random samples of the same size from the population and calculate a 95% CI for each sample, approximately 95% of these intervals would include the true population mean.

# **7. Comparing Species Groups**

Group by species, calculate summary statistics and confidence intervals

- Group the dataset by the species column and calculate the mean and standard deviation of flipper length for each species.
        Question: What differences do you observe between species in terms of mean flipper length? How do they compare to the standard deviations?
- Take a sample of 15 observations per species and compute a 90% confidence interval for the mean flipper length within each group.
        Question: Interpret the confidence intervals. Do they overlap between species? What does this suggest about the differences in flipper length across species?
"""

#grouping by species, for flipper lenght and computing mean, std dev
data_species = (data.groupby("species")['flipper_length_mm'].agg(["mean", "std","count"]))
data_species

"""**Comments**:

On average, Gentoo penguins have the longest flippers, Adelie the shortest, Chinstrap fall in between.

The std deviations are similar, they fall in an interval of ~6.52 and ~7.13 mm. -> similar variability within species for this variable.

However, the differences between the means of different species are wider (for this variable).
Ex:
- chinstrap mean - adelie mean = (195.8 - 190) ~ 5.8
- gentoo mean - chinstrap mean = (217-195.8 ) = 21,2

This suggests that the species differ significantly for the flipper length.
"""

#samples of 15 individual per species , computing 90% confidence intervals

#random sampling 15 penguins
from scipy import stats
sample_15 = (data.groupby("species").sample(n=15, replace=False, random_state = 100))

#confidence intervals
CIs = []

for species, group in sample_15.groupby("species"):
  # x is the group of flipper lengths in the sample
  x = group["flipper_length_mm"].dropna() #getting rid of null values
  n = len(x)
  mean = x.mean()
  sdev = x.std(ddof=1)
  se = sdev/np.sqrt(n)

  t = stats.t.ppf(0.95, df=n-1) #-1 degree of freedom

  L = mean - t *se
  U = mean+t * se

  CIs.append({"species": species, "sample_mean": mean, "lower_90_ci": L, "upper_90_ci":U})

#converting to data frame
CIs_df = pd.DataFrame(CIs)
CIs_df

"""**Comments**:

1. There is no overlap between the CI values of Adelie and Gentoo and between Chinstrap and Gentoo
2. Adelie and Chinstrap don't overlap by just little distance: Upper_Adelie = 191.95, Lower_Chin = 193.89

This suggests that Gentoo penguins' flippers are much longer than those of the other 2 species, in general, while Adelie and Chinstrap have a difference of ~8 mm.

## 8. To go further
- Question: If you were to extend this analysis, what other questions would you investigate ? You’re not expected to actually investigate them, just to come up with ideas that can be explored using this dataset.

We plan to:
1. Using the dataset we could build a test, training and validation set for a supervised learning model -> later use for penguin classification
2. Going deeper into statistical analysis:
    - test whether the difference in numerical variables is significant with the *p-value*
    - create correlation heatmaps to test the relationship between all the continuous variables
"""