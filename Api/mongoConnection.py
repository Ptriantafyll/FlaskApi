import os
from pymongo import MongoClient
from dotenv import load_dotenv
from models import user

load_dotenv() # load .env file

client = MongoClient("mongodb+srv://PanosTriantafyllopoulos:" + os.getenv("MONGODB_ATLAS_PW") +"@url-rating.jdbbkmz.mongodb.net/?retryWrites=true&w=majority&authSource=admin")
db = client['url-rating']

try:
    db.create_collection("user")
except Exception as e:
    print(e)

user_validator = user.validator()
db.command("collMod", "user", validator=user_validator)

users = [
    {
        "links": [
            {
                "url": "link1",
                "rating": 3
            },
            {
                "url": "link2",
                "rating": 3
            },
            {
                "url": "link3",
                "rating": 3
            }
        ]
    },
    {
        "links": [
            {
                "url": "aaa",
                "rating": 3
            },
            {
                "url": "bbb",
                "rating": 3
            },
            {
                "url": "ccc",
                "rating": 3
            }
        ]
    }
]

inserted_ids = db.get_collection('user').insert_many(users).inserted_ids
print(inserted_ids)

def connect_to_db():
    client = MongoClient("mongodb+srv://PanosTriantafyllopoulos:" + os.getenv("MONGODB_ATLAS_PW") +"@url-rating.jdbbkmz.mongodb.net/?retryWrites=true&w=majority")
    db = client['url-rating']
    return db
