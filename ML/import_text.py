
# ? This file will import the text of all websites rated into mongodb url collection
import time
import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
import mongoDB_connection


def validator():
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "properties": {
                "_id": {
                    "bsonType": "objectId"
                },
                "url": {
                    "bsonType": "string",
                    "pattern": "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
                },
                "text": {
                    "bsonType": "string",
                    "description": "Text of the website",
                },
                "language": {
                    "bsonType": "string",
                    "description": "Language of the page"
                }
            }
        }
    }


# todo add the texts of all websites to mongodb
mongoDB_connection.connect_to_mongodb()
db = mongoDB_connection.db
db.command("collMod", "url", validator=validator())

counter = 0
not_allowed = {}
urls = set()
start_time = time.time()
# ? iterate users
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
                # html_content = response.text
                print(response.status_code)
            else:
                counter = counter + 1
                not_allowed[url] = response.status_code
                print(response.status_code)
        except Exception as e:
            print("An exception occurred: ", type(e))
            not_allowed[url] = str(type(e))
            continue
end_time = time.time()

print("Not allowed by requests: " + str(counter))
# print(not_allowed)
# for website, status in not_allowed.items():
#     print(website, ": ", status)

# Specify the file path where you want to save the JSON data
file_path = r"C:\Users\ptria\source\repos\FlaskApi\ML\not_allowed.json"

# Write the dictionary to the JSON file
with open(file_path, 'w') as json_file:
    json.dump(not_allowed, json_file)

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

# ? with beautifulsoup
# url = "https://el.wikipedia.org/wiki/%CE%9B%CE%B9%CE%BF%CE%BD%CE%AD%CE%BB_%CE%9C%CE%AD%CF%83%CE%B9"
# url = "https://www.electronics-tutorials.ws/accircuits/complex-numbers.html"

# response = requests.get(url)
# try:
#     response = requests.get(url)
# except requests.exceptions.TooManyRedirects as e:
#     print(e)
# print(response.status_code)

# Check if the request was successful
# if response.status_code == 200:
#     html_content = response.text
# else:
#     print(
#         f"Failed to retrieve the website. Status code: {response.status_code}")

# print(html_content)


# soup = BeautifulSoup(html_content, 'html.parser')
# # soup2 = BeautifulSoup(html_content, 'html5lib')

# # for script in soup2(['script', 'style', 'noscript']):
# # script.extract()

#  Extract the text from the BeautifulSoup object
# website_text = soup.get_text()
# # website_text2 = soup2.get_text()
# print(website_text)
# # print(website_text2)

#  Get the language of the page
# html_tag = soup.find('html')

# if html_tag and 'lang' in html_tag.attrs:
#     language = html_tag['lang']
#     print(f"The language of the page is: {language}")
# else:
#     print("Language not specified.")

# ? Get text with selenium
# chrome_options = Options()
# chrome_options.add_argument('--no-sandbox')
# chrome_options.add_argument('--headless')
# chrome_options.add_argument('--disable-dev-shm-usage')
# driver = webdriver.Chrome(options=chrome_options)

# driver.maximize_window()

# # url = "https://el.wikipedia.org/wiki/%CE%9B%CE%B9%CE%BF%CE%BD%CE%AD%CE%BB_%CE%9C%CE%AD%CF%83%CE%B9"
# url = "https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/qatar2022"
# driver.get(url)
# website_text = driver.find_element(By.XPATH, "/html/body").text
# print(website_text)

# ? for websites throwing FLOC error
# options = uc.ChromeOptions()
# options.add_argument("--headless=new")
# driver = uc.Chrome(options=options)

# url = "https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/qatar2022"
# driver.get(url)
# time.sleep(5)

# website_text = driver.find_element(By.XPATH, "/html/body").text
# print(website_text)
# driver.quit()
