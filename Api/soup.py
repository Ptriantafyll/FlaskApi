import requests
from bs4 import BeautifulSoup

r = requests.get("https://en.wikipedia.org/wiki/2022_FIFA_World_Cup")
soup = BeautifulSoup(r.text, "html.parser")

for link in soup.find_all("a"):
    print(link)
