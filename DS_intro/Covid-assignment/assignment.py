## NB: the original assignment was submitted as colab notebook, therefore the code is suitable for that format
#importing necessary modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',100)
plt.rcParams['figure.figsize']= (12,8)

continent_colors = {'Africa': '#1f77b4', 'Asia': '#ff7f0e', 'Europe': '#2ca02c',
'North America': '#d62728', 'South America': '#9467bd', 'Oceania': '#8c564b'
}

url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
df = pd.read_csv(url)
df.head()

print(df.shape)

df['date'] = pd.to_datetime(df['date'])


n_null = df.isna().sum()
print(n_null)

for col in df.columns:
  tot = len(df)
  miss = df[col].isna().sum()
  missing_perc = miss/tot*100
  print(f"{col}: {missing_perc:.2f}% missing")

'''
I see that there are several columns which lack the majority of the data.
For now i won't remove null values and I will just do selective cleaning for each step of the analysis I perform.
'''

plt.figure(figsize=(12,3))
sns.heatmap(df.isna().mean().to_frame().T, cbar = False)
plt.title("MIssing values per column")

'''
In this case I decided to use a single column heatmap 
because column that are mostly complete or mostly emply are easily visible.
To have the single column df i use to_frame as param of sns.heatmap.

Cases and deaths lack 4-5% of the data , so I can start from here.
'''

print(df['total_cases'].describe())
print(df['total_cases'].notna().describe())

'''
total cases: mean = 7.3 M; max = 775 M
i can guess few countries are contributing
with many cases -> right skewed distributiin
'''
print(df['total_cases'].median())
print(df['total_cases_per_million'].median())

'''
I want to look at the distribution of cases per continent.
I use the boxplot because i have categorical (continent) 
vs numeric (cases) data and within the categorical there
 are multiple outbreaks (different countries).
'''
df_cl = df.dropna(subset=["continent","total_cases_per_million"])

#boxplot
plt.figure(figsize=(14,8))
sns.boxplot(df_cl, x='continent', y='total_cases_per_million', hue='continent', showfliers=False, palette= continent_colors, dodge=False)
plt.title("Tot cases per million, by continent")
plt.xlabel('Continent')
plt.ylabel("Tot cases per million")
plt.tight_layout()

'''
- most countries have **low "per million" cases**: their median is low compared to the max value;
- Europe, Oceania and North america have very **tall boxes**; this suggests **high variability among the continent's countries**
- Europe has the highest median -> higher case rate
'''

g = sns.FacetGrid(df_cl, col='continent', col_wrap=3, height=4, sharex=False, sharey=False)
g.map_dataframe(sns.histplot, x='total_cases_per_million', hue = 'continent', palette=continent_colors, bins=50)
g.set_axis_labels("Total Cases per Million", "Count")
g.set_titles("{col_name}")
plt.xscale('log')
plt.tight_layout()
plt.show()

'''
I tried to make an histogram to see total cases per mil **density** but it was unreadable:
- x range: 0> 763598;
- all countries are overlayed and smaller values are not visible.

So I switched to separate facets.
The histograms are still a bit difficult to read but with logaritmic scale on X-axis the extremes should be compressed.

**Europe and South America seem to have the "wider, thicker" tail**.


Now i would like to see more in detail the countries that were most impacted , maybe the top 10.
'''

cols = [
    "location", "date",
    "total_cases", "total_deaths",
    "total_cases_per_million", "total_deaths_per_million"
]
df_dc = df[cols]

exclude = [
    'World', 'Africa', 'Asia', 'Europe', 'European Union',
    'International', 'North America', 'Oceania', 'South America'
]
df = df[~df["location"].isin(exclude)]

latest = df.sort_values("date").groupby("location", as_index=False).tail(1)

latest = latest.dropna(subset=["total_cases", "total_deaths"])

latest["cfr"] = (latest["total_deaths"] / latest["total_cases"]) * 100

top_cases = latest.sort_values("total_cases", ascending=False).head(15)[
    ["location", "total_cases", "total_deaths", "cfr"]
]
print("\nTop 15 countries by total cases:", top_cases.to_string(index=False))


top_deaths = latest.sort_values("total_deaths", ascending=False).head(15)[
    ["location", "total_deaths", "total_cases", "cfr"]
]
print("\nTop 15 countries by total deaths:", top_deaths.to_string(index=False))


top_cfr = latest[latest["total_deaths"] > 1000].sort_values("cfr", ascending=False).head(15)[
    ["location", "cfr", "total_cases", "total_deaths"]
]
print("\nTop 15 countries by CFR (deaths per 100 cases):")
print(top_cfr.to_string(index=False))

'''
I can see that some of the most impacted countries by deaths and cases were Brazil, USA, Germany, France, UK, Russia.
When i measure the CFR I see other names: they probably had more deaths per cases but less deaths and cases overall; probably connected to the population dimension as well.
'''

df_cont = df[df["continent"].notna()]
df_latest = df_cont.sort_values("date").groupby("location", as_index=False).tail(1)
df_latest["cfr"] = df_latest["total_deaths"] / df_latest["total_cases"]

#handling missing values and avoiding division by 0
df_latest["cfr"] = df_latest["cfr"].fillna(0)
df_latest.loc[~df_latest["cfr"].apply(np.isfinite), "cfr"] = 0

cfr_continent = (
    df_latest.groupby("continent", as_index=False)["cfr"]
    .mean()
    .sort_values("cfr", ascending=False)
) #avg cfr by continent

plt.figure(figsize=(8, 5))
sns.barplot(data=cfr_continent, x="continent", y="cfr", palette="muted")

plt.title("Average cfr by continent", fontsize=14)
plt.xlabel("Continent", fontsize=12)
plt.ylabel("Average CFR", fontsize=12)
plt.tight_layout()
plt.show()

'''
Seems like the highes cfr were in south america and africa, 
even if North america and Europe also had countries with many deaths.
The barplot is more appropriate in this case because i 
can distinguish the continents (qualitative) and clearly see the 
different height of the bars (quantitative).
I'm not exaclty interested in the mean nor median now.
'''

df_hdi = df_latest[['location','continent','human_development_index', 'gdp_per_capita']].dropna()
correl = df_hdi['human_development_index'].corr(df['gdp_per_capita'])
print(correl)

'''
I see a ~0.754 correlation between HDI and GDP so i feel confident 
that HDI is quite reliable to understand countries wealth
'''

def hdi_tier(hdi):
  if hdi < 0.55:
    return 'low'
  elif hdi < 0.70:
    return 'medium'
  elif hdi < 0.80:
    return 'high'
  else:
    return 'very high'


#sort the values in ascending order
hdi_byc = (df_latest[['continent','human_development_index']].dropna().groupby(
    'continent')['human_development_index'].mean().sort_values(
    ascending=False))

print("\nHDI by continent", hdi_byc)


#aggregation of HDi and CFR by continent
stats = (df_latest.groupby("continent", as_index=False).agg({"cfr":"mean", "human_development_index":"mean"}).sort_values(by= 'cfr', ascending = False))
print(stats)

#visualising this correlation
plt.figure(figsize=(8, 6))
sns.scatterplot(data=stats, x="human_development_index", y="cfr", hue="continent",
    s=150)

#regression line for more clarity
sns.regplot( data=stats,
    x="human_development_index", y="cfr",
    scatter=False, color="black",
    line_kws={"linewidth": 1, "linestyle": "--"}
)

plt.title("Correlation between average cfr and hdi, by continent", fontsize=14)
plt.xlabel("average hdi", fontsize=12)
plt.ylabel("average cfr", fontsize=12)
plt.tight_layout()
plt.show()

'''
I think this plot was the right choice because we are dealing 
with 2 quantitative variables and i have to map the average 
correlation for each country. The regression line represents 
the best model to explain the relationship between HDI and CFR (average) by continent.
I see a slightly **negative** correlation, which could make sense.
'''

corr_hdi_cfr = stats['human_development_index'].corr(stats['cfr'])
print(corr_hdi_cfr)

'''
this calculation confirms the slighly negative correlation.

Now i want to summerise the correlation between some demographics and socio-economic factors, deaths and cases.
To do this, I will create an heatmap.
'''

feats = ["life_expectancy", "human_development_index", "aged_65_older",
         "people_vaccinated_per_hundred"]
keys = ["total_cases_per_million" , "total_deaths_per_million"]
cols = feats + keys

#i decided to impute missing values using the median
df_corr = df[cols].fillna(df[cols].median())

#i found pearson to be a good method for correlation in this case
corr_matrix = df_corr.corr()

corr_subset = corr_matrix.loc[feats, ["total_deaths_per_million", "total_cases_per_million"]]

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation between deaths, cases and indicators")  # <- use ax.set_title()
plt.show()
'''
I can see some correlation especially with age, hdi and life expectancy but not much with people vaccinated.

Since some hospitals where one of the most critical spaces, I am interested in observing the burden by continent.
The data should only be about the hospitalisations for COVID, which took some hospital beds from other health issues.
'''

df_hosp = df[['date','location','continent','hosp_patients_per_million']].dropna()
print(df_hosp.head())
print(df_hosp.shape)

'''
I can see that i lost some entries that lack the metrics i chose, but it is still a robust amount compared to the beginning.

I will use a **boxplot** to visualise the hospitalised patients per million by continent. This way I can alsio get an idea of the medians and the sparsity (variance)
'''

fig, ax = plt.subplots(figsize=(10,6))
#plt.figure(figsize=(10, 6))
sns.boxplot(data=df_hosp, x='continent', y='hosp_patients_per_million')
ax.set_title("Hospitalization burden by continent")
ax.set_xlabel("Continent")
ax.set_ylabel("Hospitalized Patients per Million")
#plt.yscale('log')
plt.show()

'''
- The medians seem to be more or less similar, but Europe is the area with more variance.
- I remember South america and Africa had the highest CFR
- here I see South Americs's median for hospidalised patient is the lowest.

Next up, I will visualise the hospitalisation trends over time. I remember the breakout was around the spring of 2020, so I expect massive hospitalisation in the winter of 2020 (the following winter). I also know in the past 2-3 years there have been other covid cases so maybe i will see 2 peaks.
'''

g_trend = df_hosp.groupby('date')['hosp_patients_per_million'].mean().reset_index()
fig, ax = plt.subplots(figsize=(12,6))
sns.lineplot(data=g_trend, x='date', y='hosp_patients_per_million', color='blue')
ax.set_title("Global Hospitalization Trend Over Time (per million people)")
ax.set_xlabel("Date")
ax.set_ylabel("Hospitalized Patients per Million")
ax.grid(True)
plt.show()

'''
I see the most important peaks in the winter of 2020-2021 
(as expected), winter of 2021-2022 (maybe the coincidence with
 winter made people more vulneranle).
 '''