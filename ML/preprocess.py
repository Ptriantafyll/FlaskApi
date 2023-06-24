
# ? data preprocessing in this file

import pandas as pd
import json


df = pd.read_csv("data.csv")
# this returns a string with value: array of urls with their ratings
my_string = df["links"][1]
print(type(my_string))

# ? convert string to json object - list of dictionarys in python
my_json_object = json.loads(my_string)
print(type(my_json_object))

# ? how to access urls and ratings
print(my_json_object[0]["url"])
print(my_json_object[0]["rating"])

# todo: have selnium access the links text
# todo: feature extraction from words to vector?
# todo: create tf-idf vectors
# todo: word embedding gensim and bert
