def validator():
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "properties": {
                "_id": {
                    "bsonType": "objectId"
                },
                "url": {
                    "bsonType": "string",
                    "pattern": "\b(?:(?:https?|ftp):\/\/|www\.)[-a-zA-Z0-9+&@#\/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#\/%=~_|]"
                },
                "text": {
                    "bsonType": "string",
                    "description": "text of the website",
                }
            }
        }
    }
