
# ? This file will import the text of all websites rated into mongodb url collection
import time
import json
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc
import mongoDB_connection

mongoDB_connection.connect_to_mongodb()
db = mongoDB_connection.db

counter = 0
not_allowed, allowed = {}, {}
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
                allowed[url] = response.status_code
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
not_allowed_path = r"C:\Users\ptria\source\repos\FlaskApi\ML\json\not_allowed.json"
allowed_path = r"C:\Users\ptria\source\repos\FlaskApi\ML\json\allowed.json"

# Write the dictionary to the JSON file
with open(not_allowed_path, 'w') as json_file:
    json.dump(not_allowed, json_file)

with open(allowed_path, 'w') as json_file:
    json.dump(allowed, json_file)

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")
