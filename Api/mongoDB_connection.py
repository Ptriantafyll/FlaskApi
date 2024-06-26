import os
from pymongo import MongoClient
from dotenv import load_dotenv
import certifi

def connect_to_mongodb():
    load_dotenv()  # load .env file
    client = MongoClient("mongodb+srv://PanosTriantafyllopoulos:" + os.getenv("MONGODB_ATLAS_PW") +
                         "@url-rating.jdbbkmz.mongodb.net/?retryWrites=true&w=majority&authSource=admin", tlsCAFile=certifi.where())
    global db
    db = client['url-rating']
