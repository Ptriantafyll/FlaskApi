from controllers import user as user_controller
from flask_restful import Resource


class CreateUser(Resource):
    def post(self):
        message = user_controller.create_user()
        # todo: return meaningful message
        return {"message": message}
