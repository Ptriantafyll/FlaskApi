
# ? This file will import the text of all websites rated into mongodb url collection
# import time
import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
import mongoDB_connection


mongoDB_connection.connect_to_mongodb()
db = mongoDB_connection.db


# ? Gets all links from mongodb
# ? Every url that returns a response with status code 200 goes to -> allowed.json file
# ? Every other url (with response code != 200 or with errors thrown) -> not_allowed.json
def urls_with_res_status_200_to_json():
    not_allowed, allowed = {}, {}
    urls = set()
    users = db['user'].find()
    for user in users:
        print(user["_id"])
        for link in user["links"]:
            print(link["url"])
            url = link["url"]

            if url in urls:
                print("skipping ", url, " as it already exists")
                continue
            else:
                urls.add(url)

            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    allowed[url] = response.status_code
                else:
                    continue
            except Exception as e:
                continue


# ? Gets links from a json file
# ? all urls with specified language -> specified.json
# ? all urls with not specified language -> not_specified.json
def urls_language_to_json(file_path):

    f = open(file_path)
    urls = json.load(f)
    specified, not_specified, errors = {}, {}, {}

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')

            # ? Get the language of the page
            html_tag = soup.find('html')
            if html_tag and 'lang' in html_tag.attrs:
                language = html_tag['lang']
                specified[url] = language

            else:
                print("The language of the page ", url, "is not specified")

        except Exception as e:
            print(url, " ", type(e))

    specified_path = r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\specified.json"
    not_specified_path = r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\not_specified.json"
    errors_path = r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\errors.json"
    with open(specified_path, 'w') as json_file:
        json.dump(specified, json_file)

    with open(not_specified_path, 'w') as json_file:
        json.dump(not_specified, json_file)

    with open(errors_path, 'w') as json_file:
        json.dump(errors, json_file)


def urls_not_allowed_by_bs4_to_json(file_path):
    f = open(file_path)
    urls = json.load(f)

# urls_with_res_status_200_to_json(
    # r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\allowed.json")

# urls_language_to_json(
    # r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\specified.json")
