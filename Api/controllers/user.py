from bson.objectid import ObjectId
from models import user
import mongoDB_connection
import tensorflow as tf
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import numpy as np
from transformers import TFDistilBertForSequenceClassification
from transformers import AutoTokenizer
import undetected_chromedriver as uc

import requests
from bs4 import BeautifulSoup


# Returns the text of a given website
def get_website_text(url):
    # Step 1: Fetch HTML content
    response = requests.get(url, timeout=(5, 5))
    html_content = response.content

    # Step 2: Parse HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Step 3: Extract text using get_text()
    text_content = soup.get_text()

    return text_content


# This is needed to import a function from different directory
import sys
sys.path.append(r"C:\Users\ptria\source\repos\FlaskApi\Api")
from functions import preprocess

# ? Load pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")
# ? distilbert base model with pretrained weights
pretrained_bert = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")


# Returns what action to take when a user adds a rating (add if not exists, update if exists)
def handle_filters_for_new_rating(userId, current_user, linkToUpdate):
    num_of_links = 0
    if "links" in current_user:
        # ? user has links
        links = current_user.get("links")
        urls = [link['url'] for link in links]

        if linkToUpdate["url"] in urls:
            # ? link has been rated before
            filters = {"_id": userId, "links.url": linkToUpdate["url"]}
            updates = {"$set": {"links.$.rating": linkToUpdate["rating"]}}
            num_of_links = len(links)
        else:
            # ? link has not been rated before
            new_link = {"url": linkToUpdate["url"],
                        "rating": linkToUpdate["rating"]}

            filters = {"_id": userId}
            updates = {"$push": {"links": new_link}}
            num_of_links = len(links) + 1
    else:
        # ? user has no links
        filters = {"_id": userId}
        updates = {"$set": {"links": [linkToUpdate]}}
        num_of_links = 1

    return filters, updates, num_of_links


# Initializes MongoDB cluster
def initialize_db_for_user_collection():
    db = mongoDB_connection.db
    user_validator = user.validator()
    db.command("collMod", "user", validator=user_validator)

    return db


# Inserts a new User into the MongoDB cluster
def create_user():
    db = initialize_db_for_user_collection()

    # ? create collection 'user' if it does not exist
    try:
        db.create_collection("user")
    except Exception as e:
        print(e)  # todo: probably will get rid of this print

    # ? create a new user (empty for now as _id is generated automatically) and add to db
    newUser = db.get_collection("user").insert_one({})
    newUserId = str(newUser.inserted_id)

    return newUserId


# Handles the case that a User has selected a Rating
def user_adds_rating(userToUpdate, linkToUpdate):
    db = initialize_db_for_user_collection()

    userId = ObjectId(userToUpdate)
    current_user = db.get_collection("user").find_one({"_id": userId})
    filters, updates, num_of_links = handle_filters_for_new_rating(
        userId, current_user, linkToUpdate)

    db.get_collection("user").update_one(filters, updates)

    return num_of_links


# Initializes chrome driver using (undetected) selenium
def initialize_chrome_with_selenium():
    # chrome_options = Options()
    # chrome_options.add_argument('--no-sandbox')
    # chrome_options.add_argument('--headless')
    # chrome_options.add_argument('--disable-dev-shm-usage')
    # driver = webdriver.Chrome(chrome_options=chrome_options)

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = uc.Chrome(options=options)
    driver.maximize_window()

    return driver


# Tokenizes text and makes it into a structure that bert can use
def tokenize_text(website_text, links_to_rate):
    input_ids = np.zeros(
        (len(links_to_rate), 128))
    attention_masks = np.zeros((len(links_to_rate), 128))

    website_text = preprocess.clean_text(website_text)
    website_text = website_text.lower()
    website_text = preprocess.filter_sentences_english_and_greek(
        website_text)

    # encode text
    tokenized_text = tokenizer.encode_plus(
        website_text,
        max_length=128,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    input_ids = tokenized_text.input_ids
    attention_masks = tokenized_text.attention_mask

    return input_ids, attention_masks


# Predict rating of a usl for the current_user user
def predict_rating(user_model, input_ids, attention_masks, ratings, url):
    prediction = user_model.predict([input_ids, attention_masks])
    predicted_class = [max(0, min(round(x), 4))
                       for x in prediction.flatten()]
    predicted_rating = predicted_class[0]
    ratings[url] = predicted_rating


def get_website_text_if_exists(driver, url):
    driver.get(url)
    time.sleep(5)

    timeout = 10
    wait = WebDriverWait(driver, timeout)
    # ? check if website has a body
    element_present = EC.presence_of_element_located(
        (By.XPATH, "/html/body"))

    wait.until(element_present)
    website_text = driver.find_element(By.XPATH, "/html/body").text
    website_text = get_website_text(url)

    return website_text



# Use the model for a user and predict the ratings they would give to given websites
def get_ratings_for_user(str_userId, links_to_rate):
    db = initialize_db_for_user_collection()

    # userId = ObjectId(str_userId)
    # TODO: have models for all users and load the model based on the userId
    user_model = tf.keras.models.load_model(
        r"C:\Users\ptria\source\repos\FlaskApi\rating_model")


    ratings = {}
    for url in links_to_rate:
        # Get text of all websites (links_to_rate) using selenium
        driver = initialize_chrome_with_selenium()
        try:
            website_text = get_website_text_if_exists(driver, url)
            input_ids, attention_masks = tokenize_text(
                website_text, links_to_rate)
            predict_rating(
                user_model, input_ids, attention_masks, ratings, url)

        except Exception as e:
            print("error" + str(e))
        finally:
            driver.quit()

    return ratings


# Returns the # of urls a user has rated
def get_num_of_ratings_for_user(str_userId):
    db = initialize_db_for_user_collection()

    userId = ObjectId(str_userId)
    current_user = db.get_collection("user").find_one({"_id": userId})

    num_of_links = 0
    if "links" in current_user:
        # ? user has links
        num_of_links = len(current_user.get("links"))

    return num_of_links
