from flask import Flask
from flask_restful import Resource, Api
from pymongo import MongoClient

app = Flask(__name__)
api = Api(app)

client = MongoClient("mongodb+srv://PanosTriantafyllopoulos:1234@poidb.gojkx.mongodb.net/?retryWrites=true&w=majority")
db = client.test # test is the name of the db
# users = db['users']
# cursor = users.find({})
# for user in cursor:
#     print(user["username"])

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)