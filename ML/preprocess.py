
# ? data preprocessing in this file

import pandas as pd
import json
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
my_string = df["links"][1]
# print(type(my_string))

# ? convert string to json object - list of dictionarys in python
my_json_object = json.loads(my_string)
# print(type(my_json_object))
# print(my_json_object)

# ? how to access urls and ratings
# print(my_json_object[0]["url"])
# print(my_json_object[0]["rating"])

# todo: have selnium access the links text
chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=chrome_options)

driver.maximize_window()

url = my_json_object[0]["url"]
driver.get(url)
website_text = driver.find_element(By.XPATH, "/html/body").text
# print(website_text)
# print(type(website_text))

driver.quit()

# todo: Add url and text into mongodb to see the format - add the texts of all websites

# mongoDB_connection.connect_to_mongodb()  # todo: maybe remove that later

# db = mongoDB_connection.db
# collection = db.get_collection("url")
# collection.insert_one(
#     {
#         "url": my_json_object[0]["url"],
#         "text": website_text
#     }
# )
print(my_json_object)
# for user in my_json_object :
#     print(user)


# todo: feature extraction from words to vector?
# todo: create tf-idf vectors
# todo: word embedding gensim and bert
