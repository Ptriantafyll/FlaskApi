from controllers import user as user_controller
from flask_restful import Resource

class CreateUser(Resource):
    # todo: make post request to /newuser create a user in the database
    def post(self):
        # todo: here we should create the user
        message = user_controller.create_user()
        # todo: return meaningful message
        return {"message" : message}

