import pandas as pd
ds = pd.read_csv("authors_description.csv", header = None)
print("File read!")
#print("Head: ", ds.head())

#print(ds.info())
#print(ds.describe())
#print(ds.isnull().sum())
print(ds.shape)

"""
Now i want to create new columns
for each row i have to """

ds.columns = ["description"]

#ds["word_count"] = ds["description"].apply(lambda x: len(x.split()))

#count the words
ds["desc_split"] = ds["description"].str.split(' ')
ds["w_count"] = ds["desc_split"].apply(len)

#count the digits
ds["nospace"] = ds["description"].str.replace(" ","", regex= False)
ds["char_count"] = ds["nospace"].apply(len)

#check uppercase

# --------- PRE-PROCESSING -------
#1. LOWER CASE EVERYTHING
ds['lowercase'] = ds['description'].apply(lambda x: x.lower())

#2. remove \n
ds['lowercase_clean'] = ds['lowercase'].str.replace("\n", " ", regex= False)
#ds['lowercase_clean'] = ds['lowercase'].apply(lambda x : x.replace("\n", " "))

#3. remove punctuation & special chars
import re
ds['no_punctuation'] =  ds['lowercase_clean'].apply(lambda x: re.sub(r'[^\w\s]','', x))#matches a single character that is not word nor space


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english') # list of stop words -> not informative

# removing the stopwords'
ds['no_stops'] = ds[]
.apply(lamba x: ' '.join([word for wird in x.split() if word not in (stop)]) )
#when you do a join -> list; append -> just adds element to list

#TOKENISATION
#1: namually w split function
#2: tokeniser method from nltk
from nltk import tokenize
tokenizer = tokenize.SpaceTokenizer()
ds['tokenised']= ds['no_stops'].apply(lambda x:tokenizer.tokenize(x) )

#STEMMING
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("french", ignore_stopwords=True)
ds['stemminf']= ds['no_stops'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

print(ds.head())