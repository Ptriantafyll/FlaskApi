from bson.objectid import ObjectId
from models import user
import mongoDB_connection
import random
import tensorflow as tf
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def create_user():
    db = mongoDB_connection.db
    user_validator = user.validator()
    db.command("collMod", "user", validator=user_validator)

    # ? create collection 'user' if it does not exist
    try:
        db.create_collection("user")
    except Exception as e:
        print(e)  # todo: probably will get rid of this print

    # ? create a new user (empty for now as _id is generated automatically) and add to db
    newUser = db.get_collection("user").insert_one({})
    newUserId = str(newUser.inserted_id)

    return newUserId


def user_adds_rating(userToUpdate, linkToUpdate):
    db = mongoDB_connection.db
    user_validator = user.validator()
    db.command("collMod", "user", validator=user_validator)

    userId = ObjectId(userToUpdate)
    num_of_links = 0
    current_user = db.get_collection("user").find_one({"_id": userId})
    if "links" in current_user:
        # ? user has links
        links = current_user.get("links")
        urls = [link['url'] for link in links]

        if linkToUpdate["url"] in urls:
            # ? link exists
            filters = {"_id": userId, "links.url": linkToUpdate["url"]}
            updates = {"$set": {"links.$.rating": linkToUpdate["rating"]}}
            num_of_links = len(links)
        else:
            # ? link does not exist
            new_link = {"url": linkToUpdate["url"],
                        "rating": linkToUpdate["rating"]}
            print(new_link)
            filters = {"_id": userId}
            updates = {"$push": {"links": new_link}}
            num_of_links = len(links) + 1
    else:
        # ? user has no links
        filters = {"_id": userId}
        updates = {"$set": {"links": [linkToUpdate]}}
        num_of_links = 1

    db.get_collection("user").update_one(filters, updates)

    # todo: return meaningful message
    return num_of_links


def get_ratings_for_user(str_userId, links_to_rate):
    db = mongoDB_connection.db
    user_validator = user.validator()
    db.command("collMod", "user", validator=user_validator)

    userId = ObjectId(str_userId)
    current_user = db.get_collection("user").find_one({"_id": userId})

    # for url in links_to_rate:
        # ratings[url] = random.randint(1, 5)

    # todo: for now generate random number for each link in links_to_rate
    # todo: later make the rating based on the model of the user

    user_model = tf.keras.models.load_model(r"C:\Users\ptria\source\repos\FlaskApi\rating_model")

    ratings = {}
    for url in links_to_rate:
        # todo: Step 1 get text of all websites (links_to_rate) using selenium

        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(chrome_options=chrome_options)

        driver.maximize_window()
        try:
            driver.get(url)
            time.sleep(5)

            timeout = 10
            wait = WebDriverWait(driver, timeout)
            # ? check if website has a body
            element_present = EC.presence_of_element_located(
                (By.XPATH, "/html/body"))

            wait.until(element_present)
            print(url, " loaded successfully!")
            website_text = driver.find_element(By.XPATH, "/html/body").text
            print("got text of: ", url)

            # todo: Step 2 tokenize text and make it into a structure that bert can use

            # todo: Step 3 user_model.predict
            # todo: Step 4 ratings[url] = prediction rounded,capped
        except Exception as e:
            print("error" + str(e))
        finally:
            print("quitting driver")
            driver.quit()


    return ratings


def get_num_of_ratings_for_user(str_userId):
    db = mongoDB_connection.db
    user_validator = user.validator()
    db.command("collMod", "user", validator=user_validator)

    userId = ObjectId(str_userId)
    current_user = db.get_collection("user").find_one({"_id": userId})

    num_of_links = 0
    if "links" in current_user:
        # ? user has links
        num_of_links = len(current_user.get("links"))

    return num_of_links
