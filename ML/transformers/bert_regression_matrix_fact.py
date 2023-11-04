import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import TFBertModel
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
import json

# Load pretrained tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

# This is needed to import a function from different directory
import sys
sys.path.append(r"C:\Users\ptria\source\repos\FlaskApi\ML")
import matrix_factorization

# Returns user-url matrix as pandas DataFrame 
df = matrix_factorization.perform_martix_factorization()

# ? File that contains all the users in the mongodb cluster
# user_file = open(
#     r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\users.json")
# users = json.load(user_file)

# ? File that contains all the urls in the mongodb cluster
url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json", encoding="utf-8")
urls = json.load(url_file)

# pick a user from pandas df
# user = df.index[0]
user = df.index[11]

# Find max length
# max_len = 0
# for i in range(len(urls)):
#     text = urls[i]['text']
#     print(urls[i]['url'], " : ", urls[i]['language'])

#     # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
#     input_ids = tokenizer.encode(text, add_special_tokens=True)

#     # Update the maximum sentence length.
#     max_len = max(max_len, len(input_ids))

# todo: if max_length > 512 -> max length = 512
# print('Max sentence length: ', max_len)

# Example of encode_plus
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


# max_length = 329295
# ? use max length 512 instead of the max length of all sequences
max_length = 512
documents_input_ids = np.zeros(
    (len(df.columns), max_length))  # ids of the bert tokenizer
# masks are 0 if they are the result of padding
documents_masks = np.zeros((len(df.columns), max_length))
# todo: change iterations and get data from pandas df
counter = 0
for url in urls:
    # encode all documents
    text = url["text"]
    tokenized_text = tokenizer.encode_plus(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    documents_input_ids[counter, :] = tokenized_text.input_ids
    documents_masks[counter, :] = tokenized_text.attention_mask
    counter = counter + 1


user_ratings = []
for url in urls:
    user_ratings.append(df.loc[user,url['url']] -1) # ratings 0-4 from 1-5

ratings = np.zeros((len(df.columns), 5))

# one-hot encoded ratings
ratings[np.arange(len(df.columns)), np.array(user_ratings)] = 1


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
dataset = dataset.shuffle(10000).batch(32, drop_remainder=True)

# Train test split
p = 0.75
train_size = int(len(df.columns)*p)

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
# Print model
rating_model.summary()
# Train model
rating_model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

rating_model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=10
)

# Save model
# rating_model.save('rating_model')

rating_model.evaluate(train_dataset, test_dataset)

# # Evaluate the model on the test data
# loss, accuracy = rating_model.evaluate(
#     documents_test, ratings_test)

# # Print the evaluation results
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)

# predictions = LSTM_model.predict(documents_test)
# predicted_classes = np.argmax(predictions, axis=1)
# print(ratings_test)
# print(predicted_classes)

# # Compute the confusion matrix
# cm = confusion_matrix(ratings_test, predicted_classes)
# print(cm)

# # Display the confusion matrix using seaborn for better visualization
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

# predictions = LSTM_model.predict(documents_train)
# predicted_classes = np.argmax(predictions, axis=1)
# print(ratings_train)
# print(predicted_classes)

# cm = confusion_matrix(ratings_train, predicted_classes)
# print(cm)

# # Display the confusion matrix using seaborn for better visualization
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()
