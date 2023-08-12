
# ? This file will import the text of all websites rated into json files
import time
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
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
        for link in user["links"]:
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
                    not_allowed[url] = response.status_code
                    continue
            except Exception as e:
                not_allowed[url] = str(e)
                continue
    allowed_path = r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\new_allowed.json"
    not_allowed_path = r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\new_not_allowed.json"

    with open(allowed_path, 'w') as json_file:
        json.dump(allowed, json_file)

    with open(not_allowed_path, 'w') as json_file:
        json.dump(not_allowed, json_file)


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
    failed, successful = {}, {}
    for url in urls:
        # ? using undetected selenium
        options = uc.ChromeOptions()
        options.add_argument("--headless=new")
        driver = uc.Chrome(options=options)

        try:
            driver.get(url)
            time.sleep(5)

            timeout = 10
            wait = WebDriverWait(driver, timeout)
            # ? check if website has a body
            element_present = EC.presence_of_element_located(
                (By.XPATH, "/html/body"))

            wait.until(element_present)
            print(url + " loaded successfully!")
            successful[url] = "success"
        except Exception as e:
            print("error" + str(e))
            failed[url] = "failed"
        finally:
            print("quitting driver")
            driver.quit()

    successful_path = r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\successful.json"
    failed_path = r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\failed.json"

    with open(successful_path, 'w') as json_file:
        json.dump(successful, json_file)

    with open(failed_path, 'w') as json_file:
        json.dump(failed, json_file)


# urls_with_res_status_200_to_json()

# urls_language_to_json(
#     r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\specified.json")


# urls_not_allowed_by_bs4_to_json(
#     r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\not_allowed.json")
