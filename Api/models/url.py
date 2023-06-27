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
                    "pattern": "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
                },
                "text": {
                    "bsonType": "string",
                    "description": "text of the website",
                }
            }
        }
    }
