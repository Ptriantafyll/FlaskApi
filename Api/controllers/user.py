from bson.objectid import ObjectId
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
        print(e)  # todo: probably will get rid of this print

    # ? create a new user (empty for now as _id is generated automatically) and add to db
    newUser = db.get_collection("user").insert_one({})
    newUserId = str(newUser.inserted_id)

    return newUserId


def user_adds_rating(userToUpdate, linkToUpdate):
    db = mongoDB_connection.db
    user_validator = user.validator()
    db.command("collMod", "user", validator=user_validator)

    from bson.objectid import ObjectId
    userId = ObjectId(userToUpdate)

    current_user = db.get_collection("user").find_one({"_id": userId})
    if "links" in current_user:
        # ? user has links
        links = current_user.get("links")
        urls = [link['url'] for link in links]

        if linkToUpdate["url"] in urls:
            # ? link exists
            filters = {"_id": userId, "links.url": linkToUpdate["url"]}
            updates = {"$set": {"links.$.rating": linkToUpdate["rating"]}}
        else:
            # ? link does not exist
            new_link = {"url": linkToUpdate["url"],
                        "rating": linkToUpdate["rating"]}
            print(new_link)
            filters = {"_id": userId}
            updates = {"$push": {"links": new_link}}
    else:
        # ? user has no links
        filters = {"_id": userId}
        updates = {"$set": {"links": [linkToUpdate]}}

    db.get_collection("user").update_one(filters, updates)

    # todo: return meaningful message
    return "User updated successfully"


def get_ratings_for_user(str_userId, links_to_rate):
    db = mongoDB_connection.db
    user_validator = user.validator()
    db.command("collMod", "user", validator=user_validator)

    userId = ObjectId(str_userId)
    current_user = db.get_collection("user").find_one({"_id": userId})
    # todo: for now generate random number for each link in links_to_rate
    # todo: later make the rating based on the model of the user
