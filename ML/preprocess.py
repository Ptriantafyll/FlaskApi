
# ? data preprocessing in this file

import spacy
import pandas as pd
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from sklearn.feature_extraction.text import TfidfVectorizer

# import sys #todo: check how to make this work for a container
# sys.path.append(
#     # probably change this to the container path to folder
#     'C:/Users/ptria/source/repos/FlaskApi/Api'
# )
import mongoDB_connection

df = pd.read_csv("ML/ratings.csv")
print(df.head())
# this returns a string with value: array of urls with their ratings
my_string = df["links"][0]
# print(type(my_string))

# ? convert string to json object - list of dictionarys in python
my_json_object = json.loads(my_string)
# print(type(my_json_object))
# print(my_json_object)

# ? how to access urls and ratings
# print(my_json_object[0]["url"])
# print(my_json_object[0]["rating"])

# ? have selnium access the links text
# todo : add a for loop that runs through every document and put everything below in it
chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=chrome_options)

driver.maximize_window()

url = my_json_object[0]["url"]
driver.get(url)
website_text = driver.find_element(By.XPATH, "/html/body").text
# print(url)
# print(website_text)
# print(type(website_text))

# ? get language of page
language_element = driver.find_element(By.TAG_NAME, "html")
language = language_element.get_attribute("lang")  # en-US / el
# print("THE LANGUAGE OF THE PAGE IS " + language)

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


# todo: feature extraction from words to vector?

# nltk.download('punkt')
# nltk.download('stopwords')

lcsentence = website_text.lower()  # ? make words lowercase

tokenizer = RegexpTokenizer(r'\w+')  # ? tokenizing and remove puunctuation
words = tokenizer.tokenize(lcsentence)
# words = [word for word in words if word.isalpha()]  # ? remove numbers

stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if not w in stop_words]
greek_stop_words = set(stopwords.words('greek'))
filtered_words = [w for w in words if not w in greek_stop_words]

if language == 'en-US':
    print("stemming")
    ps = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(ps.stem(word))

    print(stemmed_words)


if language == 'el':
    # ? lemmatizing in greek
    # Load the Greek language model in spacy
    nlp = spacy.load("el_core_news_sm")

    lemmatized_words = []
    for word in filtered_words:
        lemmatized_words.append(nlp(word)[0].lemma_)
    print(lemmatized_words)

# todo: create tf-idf vectors tfidfvectorizer from sklearn

# todo: documents = all website texts
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_vectors = tfidf_vectorizer.fit_transform(documents)
# tfidf_vectors = tfidf_vectors.toarray()
