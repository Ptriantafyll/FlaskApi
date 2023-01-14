from models import user
import mongoDB_connection


def create_user():
    db = mongoDB_connection.db
    user_validator = user.validator()
    db.command("collMod", "user", validator=user_validator)

    # ? create collection 'user' if it does not exist
    try:
        db.create_collection("user")
    except Exception as e:
        print(e) # todo: probably will get rid of this print

    # ? create a new user (empty for now as _id is generated automatically) and add to db
    newuser = {}
    db.get_collection("user").insert_one(newuser)

    return "User added successfully"

def user_adds_rating():
    db = mongoDB_connection.db
    user_validator = user.validator()
    db.command("collMod", "user", validator=user_validator)

    