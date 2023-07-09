
# ? data preprocessing in this file

import pandas as pd
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from greek_stemmer import GreekStemmer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# import sys #todo: check how to make this work for a container
# sys.path.append(
#     # probably change this to the container path to folder
#     'C:/Users/ptria/source/repos/FlaskApi/Api'
# )
import mongoDB_connection

df = pd.read_csv("ML/ratings.csv")
print(df.head())
# this returns a string with value: array of urls with their ratings
my_string = df["links"][2]
# print(type(my_string))

# ? convert string to json object - list of dictionarys in python
my_json_object = json.loads(my_string)
# print(type(my_json_object))
# print(my_json_object)

# ? how to access urls and ratings
# print(my_json_object[0]["url"])
# print(my_json_object[0]["rating"])

# ? have selnium access the links text
chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=chrome_options)

driver.maximize_window()

url = my_json_object[0]["url"]
driver.get(url)
website_text = driver.find_element(By.XPATH, "/html/body").text
print(url)
print(website_text)
# print(type(website_text))

# ? print language of page
language_element = driver.find_element(By.TAG_NAME, "html")  # Replace 'html' with the appropriate tag name or other selector
language = language_element.get_attribute("lang")
print("THE LANGUAGE OF THE PAGE IS " + language)

driver.quit()

# todo add the texts of all websites to mongodb

# mongoDB_connection.connect_to_mongodb()  # todo: maybe remove that later

# db = mongoDB_connection.db
# collection = db.get_collection("url")
# collection.insert_one(
#     {
#         "url": my_json_object[0]["url"],
#         "text": website_text
#     }
# )


# print(my_json_object)
# for user in my_json_object :
#     print(user)


# todo: feature extraction from words to vector?

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

print("making lowercase")
lcsentence = website_text.lower()
print(lcsentence)

# print("tokenizing")
# words = word_tokenize(lcsentence)  # ? words to tokens
# print(words)

print("reg exp tokenizing")
tokenizer = RegexpTokenizer(r'\w+')  # ? remove punctuation
words = tokenizer.tokenize(lcsentence)
words = [word for word in words if word.isalpha()]  # ? remove numbers

print("printing stop words")
stop_words = set(stopwords.words('english'))
print(stop_words)

# ! will need that for greek websites
# print("printing stop words greek")
# stop_words = set(stopwords.words('greek'))
# print(stop_words)

print("removing stop words")
filtered_words = [w for w in words if not w in stop_words]
# todo: probably add greek stop word removal
# print("before")
# print(words)
# print("after")
print(filtered_words)

# todo: if language = english -> stemming, else lemmatize in greek with spacy
print("stemming")
ps = PorterStemmer()
stemmed_words = []  
for word in filtered_words:
    stemmed_words.append(ps.stem(word))

print(stemmed_words)


# αν κάνω στεμμινγκ όχι λιματιζινγκ
# print("lemmatizing")
# lemmatizer = WordNetLemmatizer()
# lemmatized_words = []  
# for word in filtered_words:
#     lemmatized_words.append(lemmatizer.lemmatize(word))

# print(lemmatized_words)

#? lemmatizing in greek
# import spacy

# # Load the Greek language model in spacy
# nlp = spacy.load("el_core_news_sm")

# # Text to be lemmatized
# text = "Τα παιδιά παίζουν στο πάρκο και διασκεδάζουν."

# # Tokenize the text
# doc = nlp(text)

# # Lemmatize each token
# lemmas = [token.lemma_ for token in doc]

# # Print the lemmas
# print(lemmas)

# todo: create tf-idf vectors tfidfvectorizer from sklearn
# todo: word embedding bert
# todo: word embedding gensim




