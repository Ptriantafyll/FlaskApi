from flask import Flask, request
from flask_restful import Resource, Api, reqparse


app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {"message": "Hello world"}

class KeepUser(Resource):
    # todo: make post request to /newuser create a user in the database
    def get(self):
        return {"new": "user"}
    def post(self):
        # args = parser.parse_args()
        # print(request.json)
        # todo: here we should create the user
        return {"post" : "request"}

# parser = reqparse.RequestParser()
# parser.add_argument("data", type=int, help="my test data") # parses 'data' argument (needs to be int)

# adding resources
api.add_resource(HelloWorld, '/')
api.add_resource(KeepUser, "/newuser")

if __name__ == '__main__':
    app.run(debug=True)
