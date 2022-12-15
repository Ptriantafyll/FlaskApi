def validator():
    # todo: create regex for links
    return {
        "$jsonSchema":{
            "bsonType": "object",
            "properties": {
                "links": {
                    "bsonType": "array",
                    "items": {
                        "bsonType": "object",
                        "required": ["url", "rating"],
                        "properties": {
                            "url": {
                                "bsonType": "string"
                            },
                            "rating": {
                                "enum": [1,2,3,4,5]
                            }
                        }
                    },
                    "description": "urls and ratings for them given by the user"
                }
            }
        }
    }
