def validator():
    return {
        "$jsonSchema": {
            "bsonType": "object",
            "properties": {
                "_id": {
                    "bsonType": "objectId"
                },
                "links": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "object",
                        "required": ["url", "rating"],
                        "properties": {
                            "url": {
                                "bsonType": "string",
                                "pattern": "^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$"
                            },
                            "rating": {
                                "enum": [1, 2, 3, 4, 5]
                            }
                        }
                    },
                    "description": "urls and ratings for them given by the user"
                }
            }
        }
    }
