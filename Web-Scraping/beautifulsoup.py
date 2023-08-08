
from bson.objectid import ObjectId
import json
import requests
from bs4 import BeautifulSoup
import mongoDB_connection
import time


# todo add the texts of all websites to mongodb
mongoDB_connection.connect_to_mongodb()
db = mongoDB_connection.db
db.command("collMod", "url", validator=mongoDB_connection.validator())


# ? Gets urls from a json file
# ? inserts every url that has a specified language into mongodb with its text and language
def insert_urls_with_specified_language(file_path):
    f = open(file_path)
    urls = json.load(f)

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            website_text = soup.get_text()

            # ? Get the language of the page
            html_tag = soup.find('html')
            if html_tag and 'lang' in html_tag.attrs:
                language = html_tag['lang']
                new_link = {"url": url, "text": website_text,
                            "language": language}
                db.get_collection("url").insert_one(new_link)
            else:
                print("The language of the page ", url, "is not specified")

        except Exception as e:
            print(url, " ", type(e))


# ? gets urls and their languages from a json file
# ? inserts the url with its text and language into mongodb
def insert_urls_with_lang_from_json_file(file_path):
    f = open(file_path)
    urls = json.load(f)

    for url, language in urls.items():
        try:
            # ? get the text of the page
            response = requests.get(url, timeout=10)
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')
            website_text = soup.get_text()

            new_link = {"url": url, "text": website_text,
                        "language": language}
            db.get_collection("url").insert_one(new_link)
            print("inserted")
        except Exception as e:
            print(url, " ", type(e))


# insert_urls_with_specified_language(
    # r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\specified.json")
# insert_urls_with_lang_from_json_file(
    # r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\not_specified.json")

# ? Scraping with beautifulsoup

url = "https://chat.openai.com/"
print(url)
# start_time = time.time()
response = requests.get(url)
# end_time = time.time()
print(response.status_code)
# print("time: ", end_time - start_time)

# Check if the request was successful
if response.status_code == 200:
    html_content = response.text
else:
    print(
        f"Failed to retrieve the website. Status code: {response.status_code}")

soup = BeautifulSoup(html_content, 'html.parser')
website_text = soup.get_text()
print(website_text)


# ? Get the language of the page
# html_tag = soup.find('html')

# if html_tag and 'lang' in html_tag.attrs:
#     language = html_tag['lang']
#     print(f"The language of the page is: {language}")
# else:
#     print("Language not specified.")

# soup2 = BeautifulSoup(html_content, 'html5lib')

# for script in soup2(['script', 'style', 'noscript']):
# script.extract()

#  Extract the text from the BeautifulSoup object
# website_text2 = soup2.get_text()
# print(website_text2)
