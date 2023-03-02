from controllers import user as user_controller
from flask import request
from flask_restful import Resource
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

class CreateUser(Resource):
    def post(self):
        newUserId = user_controller.create_user()
        # todo: return meaningful message
        return {"userid": newUserId}, 201


class NewRating(Resource):
    def put(self):
        data = request.json
        userToUpdate = data["user"]
        linkToUpdate = data["link"]
        message = user_controller.user_adds_rating(userToUpdate, linkToUpdate)

        # todo: return meaningful message
        return {"message": message}, 204


class GetRatings(Resource):
    def post(self, userId):
        # data = request.json
        # url = data["url"]

        # r = requests.get(url)
        # soup = BeautifulSoup(r.text, "html.parser")
        # links_of_current_page = [urljoin(url, link.get('href')) for link in soup.find_all('a')]

        # ratings = user_controller.get_ratings_for_user(
        #     userId, links_of_current_page)

        url = "https://en.wikipedia.org/wiki/FIFA_World_Cup"

        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        links_of_current_page = [urljoin(url, link.get('href')) for link in soup.find_all('a')]
        print("requests link count: " + str(len(links_of_current_page)))

        

        # driver = webdriver.Chrome('./chromedriver.exe')  # or other webdriver
        # driver.get(url)
        # links_of_current_page = [link.get_attribute('href') for link in driver.find_elements_by_tag_name('a')]
        # driver.quit()
        # print("requests link count: " + str(len(links_of_current_page)))

        driver = webdriver.Chrome()
        driver.maximize_window()
        driver.get(url)
        links_of_current_page = driver.find_elements(By.TAG_NAME, "a")
        driver.quit()
        print("SELENIUM link count: " + str(len(links_of_current_page)))
        

        
        ratings = user_controller.get_ratings_for_user(
            userId, links_of_current_page)
        
        return {"user": userId, "ratings": ratings}, 200
