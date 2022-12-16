from flask import Flask, request
from flask_restful import Resource, Api, reqparse

from routes import user as user_routes


app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {"message": "Hello world"}


# parser = reqparse.RequestParser()
# parser.add_argument("data", type=int, help="my test data") # parses 'data' argument (needs to be int)
# args = parser.parse_args()

# adding resources
api.add_resource(HelloWorld, '/')
api.add_resource(user_routes.CreateUser, '/user/newuser')


if __name__ == '__main__':
    app.run(debug=True)
