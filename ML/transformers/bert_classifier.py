import matplotlib.pyplot as plt
import seaborn as sns
from transformers import TFBertModel
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
import json

# Load pretrained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# ? File that contains all the urls in the mongodb cluster
# url_file = open(
#     r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json", encoding="utf8")
# urls = json.load(url_file)
url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls_without_errors.json", encoding="utf8")
urls = json.load(url_file)


# ? File that contains all the users in the mongodb cluster
user_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\users.json", encoding="utf8")
users = json.load(user_file)
user = users[0]


max_length = 512
documents_input_ids = np.zeros(
    (len(user['links']), max_length))  # ids of the bert tokenizer
# masks are 0 if they are the result of padding
documents_masks = np.zeros((len(user['links']), max_length))
for i in range(len(user['links'])):
    # encode all documents
    for url in urls:
        if user['links'][i]["url"] == url["url"]:
            text = url["text"]
            language = url["language"]
            break
    tokenized_text = tokenizer.encode_plus(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    documents_input_ids[i, :] = tokenized_text.input_ids
    documents_masks[i, :] = tokenized_text.attention_mask


user_ratings = []
for link in user['links']:
    user_ratings.append(link['rating'] - 1)  # ratings 0-4 from 1-5

ratings = np.zeros((len(user['links']), 5))

# one-hot encoded ratings
ratings[np.arange(len(user['links'])), np.array(user_ratings)] = 1


# use data as tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (documents_input_ids, documents_masks, ratings))


def RatingDatasetMapFunction(input_ids, attn_masks, ratings):
    return {
        'input_ids': input_ids,
        'attention_mask': attn_masks
    }, ratings


# converting to required format for tensorflow dataset
dataset = dataset.map(RatingDatasetMapFunction)


# batch size, drop any left out tensor
dataset = dataset.shuffle(10000).batch(4, drop_remainder=True)

# Train test split
p = 0.75
train_size = int(len(user['links'])*p)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# bert base model with pretrained weights
pretrained_bert = TFBertModel.from_pretrained('bert-base-multilingual-uncased')


# defining 2 input layers for input_ids and attn_masks
input_ids = tf.keras.layers.Input(
    shape=(max_length,), name='input_ids', dtype='int32')
attn_masks = tf.keras.layers.Input(
    shape=(max_length,), name='attention_mask', dtype='int32')

# 0 -> activation layer (3D), 1 -> pooled output layer (2D)
bert_embds = pretrained_bert.bert(input_ids, attention_mask=attn_masks)[1]
intermediate_layer = tf.keras.layers.Dense(
    512, activation='relu', name='intermediate_layer')(bert_embds)
output_layer = tf.keras.layers.Dense(5, activation='softmax', name='output_layer')(
    intermediate_layer)  # softmax -> calcs probs of classes

# Create model
rating_model = tf.keras.Model(
    inputs=[input_ids, attn_masks], outputs=output_layer)

for layer in rating_model.layers[:3]:
    layer.trainable = False

# Print model
rating_model.summary()
# Train model
rating_model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = rating_model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10
)

# Save model
rating_model.save('rating_model')
