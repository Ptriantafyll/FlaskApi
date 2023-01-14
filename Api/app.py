from flask import Flask
from flask_restful import Resource, Api, reqparse
import mongoDB_connection

from resources import user as user_resources

app = Flask(__name__)
api = Api(app)

mongoDB_connection.connect_to_mongodb()

# parser = reqparse.RequestParser()
# parser.add_argument("data", type=int, help="my test data") # parses 'data' argument (needs to be int)
# args = parser.parse_args()

# ? adding resources to endpoints
api.add_resource(user_resources.CreateUser, "/user/newuser")
api.add_resource(user_resources.NewRating, "/user/newrating")


if __name__ == '__main__':
    app.run(debug=True)
