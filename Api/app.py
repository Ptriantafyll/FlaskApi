import os
from flask import Flask
from flask_restful import Resource, Api
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv() # load .env file

app = Flask(__name__)
api = Api(app)

client = MongoClient("mongodb+srv://PanosTriantafyllopoulos:" + os.getenv("MONGODB_ATLAS_PW") + "@poidb.gojkx.mongodb.net/?retryWrites=true&w=majority")
# db = client.test # test is the name of the db
# users = db['users']
# cursor = users.find({})
# for user in cursor:
    # print(user["username"])

class HelloWorld(Resource):
    def get(self):
        return {"hello": "world"}

class KeepUser(Resource):
    def get(self):
        return {"new": "user"}


api.add_resource(HelloWorld, '/')
api.add_resource(KeepUser, "/newuser")

if __name__ == '__main__':
    app.run(debug=True)
