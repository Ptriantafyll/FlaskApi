from flask_cors import cross_origin
from controllers import user as user_controller
from flask import request
from flask_restful import Resource
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import undetected_chromedriver as uc
from urllib.parse import urlparse


def extract_base_urls(urls):
    base_urls = set()
    for url in urls:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
        base_urls.add(base_url)
    return base_urls


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


# Initializes chrome driver using selenium
def initialize_chrome_with_selenium(url):
    options = uc.ChromeOptions()
    options.add_argument("--headless=new")
    driver = uc.Chrome(options=options)

    driver.maximize_window()
    driver.get(url)


# Gets the urls for a website
def get_links_of_current_page(driver):
    links_of_current_page = [link.get_attribute(
        'href') for link in driver.find_elements(By.TAG_NAME, "a")]
    driver.quit()

    links_of_current_page = set(links_of_current_page)
    # links_of_current_page = extract_base_urls(links_of_current_page)
    links_of_current_page = [
        link for link in links_of_current_page if url not in link]

    return links_of_current_page


class GetRatings(Resource):
    # ? generates ratings
    @cross_origin()
    def post(self, userId):
        data = request.json
        url = data["url"]

        driver = initialize_chrome_with_selenium(url)
        links_of_current_page = get_links_of_current_page(driver)

        ratings = user_controller.get_ratings_for_user(
            userId, links_of_current_page)

        return {"user": userId, "ratings": ratings}, 200


class GetNumOfRatings(Resource):
    def get(self, userId):
        num_of_links = user_controller.get_num_of_ratings_for_user(userId)
        return {"num_of_rated_links": num_of_links}, 200
