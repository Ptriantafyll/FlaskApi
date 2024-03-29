from controllers import user as user_controller
from flask import request
from flask_restful import Resource
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


class HomePage(Resource):
    def get(self):
        return {"message": "I am an api for a browser extension"}


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
        num_of_links = user_controller.user_adds_rating(
            userToUpdate, linkToUpdate)

        # todo: return meaningful message
        return {"num_of_rated_links": num_of_links}, 200


class GetRatings(Resource):
    # ? generates ratings
    def post(self, userId):
        data = request.json
        url = data["url"]

        chrome_options = Options()
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(chrome_options=chrome_options)

        driver.maximize_window()
        driver.get(url)
        links_of_current_page = [link.get_attribute(
            'href') for link in driver.find_elements(By.TAG_NAME, "a")]
        driver.quit()

        ratings = user_controller.get_ratings_for_user(
            userId, links_of_current_page)

        return {"user": userId, "ratings": ratings}, 200


class GetNumOfRatings(Resource):
    def get(self, userId):
        num_of_links = user_controller.get_num_of_ratings_for_user(userId)
        return {"num_of_rated_links": num_of_links}, 200
