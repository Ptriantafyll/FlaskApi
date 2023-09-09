from transformers import TFBertModel
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
import json

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')


url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json", encoding="utf8")
urls = json.load(url_file)

# max_len = 0
# for i in range(len(urls)):
#     text = urls[i]['text']
#     print(urls[i]['url'], " : ", urls[i]['language'])

#     # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
#     input_ids = tokenizer.encode(text, add_special_tokens=True)

#     # Update the maximum sentence length.
#     max_len = max(max_len, len(input_ids))


# print('Max sentence length: ', max_len)

# my_text = urls[0]['text']
# token = tokenizer.encode_plus(
#     my_text,
#     max_length=329295,
#     truncation=True,
#     padding='max_length',
#     add_special_tokens=True,
#     return_tensors='tf'
# )
# print(token)
# print(token.input_ids)
# print(token.attention_mask)

user_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\users.json", encoding="utf8")
users = json.load(user_file)
user = users[0]


X_input_ids = np.zeros((len(user['links']), 329295))
X_attn_masks = np.zeros((len(user['links']), 329295))
for i in range(len(user['links'])):
    for url in urls:
        if user['links'][i]["url"] == url["url"]:
            text = url["text"]
            language = url["language"]
            break
    tokenized_text = tokenizer.encode_plus(
        text,
        max_length=329295,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    X_input_ids[i, :] = tokenized_text.input_ids
    X_attn_masks[i, :] = tokenized_text.attention_mask


user_ratings = []
for link in user['links']:
    print(link['rating'])
    user_ratings.append(link['rating'] - 1)

ratings = np.zeros((len(user['links']), 5))
print(ratings.shape)

print(len(user['links']))
print(np.array(user_ratings))
# one-hot encoded target tensor
ratings[np.arange(len(user['links'])), np.array(user_ratings)] = 1
print(ratings)


# creating a data pipeline using tensorflow dataset utility, creates batches of data for easy loading...
dataset = tf.data.Dataset.from_tensor_slices(
    (X_input_ids, X_attn_masks, ratings))
print(dataset.take(1))  # one sample data


def SentimentDatasetMapFunction(input_ids, attn_masks, ratings):
    return {
        'input_ids': input_ids,
        'attention_mask': attn_masks
    }, ratings


# converting to required format for tensorflow dataset
dataset = dataset.map(SentimentDatasetMapFunction)
print(dataset.take(1))


# batch size, drop any left out tensor
dataset = dataset.shuffle(10000).batch(2, drop_remainder=True)
print(dataset.take(1))


p = 0.75
# for each 16 batch of data we will have len(df)//16 samples, take 80% of that for train.
train_size = int(len(user['links'])*p)
print(train_size)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)


# bert base model with pretrained weights
model = TFBertModel.from_pretrained('bert-base-multilingual-uncased')


# defining 2 input layers for input_ids and attn_masks
input_ids = tf.keras.layers.Input(
    shape=(329295,), name='input_ids', dtype='int32')
attn_masks = tf.keras.layers.Input(
    shape=(329295,), name='attention_mask', dtype='int32')

# 0 -> activation layer (3D), 1 -> pooled output layer (2D)
bert_embds = model.bert(input_ids, attention_mask=attn_masks)[1]
intermediate_layer = tf.keras.layers.Dense(
    512, activation='relu', name='intermediate_layer')(bert_embds)
output_layer = tf.keras.layers.Dense(5, activation='softmax', name='output_layer')(
    intermediate_layer)  # softmax -> calcs probs of classes

sentiment_model = tf.keras.Model(
    inputs=[input_ids, attn_masks], outputs=output_layer)
sentiment_model.summary()

optim = tf.keras.optimizers.Adam()
loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

sentiment_model.compile(optimizer=optim, loss=loss_func, metrics=[acc])

hist = sentiment_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=2
)

# sentiment_model.save('sentiment_model')
