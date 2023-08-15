# FlaskApi

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Development](#development)

## Overview

An api that is connected to a MongoDB Atlas Cluster which keeps information on Users and their rating on urls.

## Requirements

- Flask
- Flask-restful
- PyMongo (for the database - mongodb atlas)
- python-dotenv (for environment variables)

You can install the requirements by running the following command in a terminal

```bash
pip install -r requirements.txt
```

If you do not have pip installed see [here](https://pip.pypa.io/en/stable/installation/) for instructions

`python -m spacy download el_core_news_sm`
`python -m spacy download en_core_web_sm`
## Development

Run the following command in a terminal and then navigate to localhost:5000 on a browser

```bash
python ./Api/app.py
```

## Usage

As of now the api accepts a post request to create a new user, a put request to update the user's ratings of websites and a post request to get the current website and calculate the ratings for the urls in it (for now the rating is random)
