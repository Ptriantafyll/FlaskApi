from controllers import user as user_controller
from flask import request
from flask_restful import Resource


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
        return {"message": message}


class GetRatings(Resource):
    def post(self):
        data = request.json
        userId = data["user"]
        links_of_current_page = data["links"]

        ratings = user_controller.get_ratings_for_user(
            userId, links_of_current_page)

        return {"ratings": ratings}
