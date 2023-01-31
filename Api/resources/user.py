from controllers import user as user_controller
from flask import request
from flask_restful import Resource
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

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
        data = request.json
        url = data["url"]

        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        links_of_current_page = [urljoin(url, link.get('href')) for link in soup.find_all('a')]

        ratings = user_controller.get_ratings_for_user(
            userId, links_of_current_page)

        return {"user": userId, "ratings": ratings}, 200
