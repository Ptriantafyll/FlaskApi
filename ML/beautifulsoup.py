
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

# f = open(r"C:\Users\ptria\source\repos\FlaskApi\ML\json\allowed.json")
f = open(r"C:\Users\ptria\source\repos\FlaskApi\ML\json\specified.json")
urls = json.load(f)
print(len(urls))
# print(urls)
counter, counter2, counter3 = 0, 0, 0
specified, not_specified, errors = {}, {}, {}
for url in urls:
    try:
        response = requests.get(url, timeout=10)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')
        # print(response.status_code)
        website_text = soup.get_text()

        # ? Get the language of the page
        html_tag = soup.find('html')
        if html_tag and 'lang' in html_tag.attrs:
            language = html_tag['lang']
            # print("The language of the page ", url, "is: ", language)
            # counter = counter + 1
            # specified[url] = language
            new_link = {"url": url, "text": website_text, "language": language}
            db.get_collection("url").insert_one(new_link)
        else:
            print("The language of the page ", url, "is not specified")
            # counter2 = counter2 + 1
            # not_specified[url] = "language not specified"

    except Exception as e:
        counter3 = counter3 + 1
        print(url, " ", type(e))
        # errors[url] = "Exception"

# print("urls with lang not specified: ", counter)
# print("urls with lang specified: ", counter2)
# print("errors: ", counter3)

# specified_path = r"C:\Users\ptria\source\repos\FlaskApi\ML\json\specified.json"
# not_specified_path = r"C:\Users\ptria\source\repos\FlaskApi\ML\json\not_specified.json"
# errors_path = r"C:\Users\ptria\source\repos\FlaskApi\ML\json\errors.json"
# with open(specified_path, 'w') as json_file:
#     json.dump(specified, json_file)

# with open(not_specified_path, 'w') as json_file:
#     json.dump(not_specified, json_file)

# with open(errors_path, 'w') as json_file:
#     json.dump(errors, json_file)

# ? Scraping with beautifulsoup

# url = "https://sci-hub.se/"
# print(url)
# start_time = time.time()
# response = requests.get(url)
# end_time = time.time()
# print(response.status_code)
# print("time: ", end_time - start_time)

# Check if the request was successful
# if response.status_code == 200:
#     html_content = response.text
# else:
#     print(
#         f"Failed to retrieve the website. Status code: {response.status_code}")

# soup = BeautifulSoup(html_content, 'html.parser')
# website_text = soup.get_text()
# print(website_text)


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
