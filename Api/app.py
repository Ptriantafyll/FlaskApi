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
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)