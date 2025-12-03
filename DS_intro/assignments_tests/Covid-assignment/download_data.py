import pandas as pd

url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
#data_set 
ds = pd.read_csv(url)


ds.to_csv("owid-covid-data.csv", index=False)

print("ok")
