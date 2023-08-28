# from gensim_word_embeddings import preprocess_document
import json
# from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
# from keras.optimizers import Adam
from keras.layers import Embedding, LSTM, Dense, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
import fasttext
import fasttext.util
import numpy as np
import tensorflow as tf
import keras

ft = fasttext.load_model(
    r"C:\Users\ptria\source\repos\FlaskApi\ML\word-embeddings\cc.el.300.bin")

# print(ft.get_nearest_neighbors("bakery"))
# print(ft.get_word_id("bakery"))

# # Step 1 Preprocess user's ratings
user_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\users.json")
users = json.load(user_file)

# # todo ? cluster of users
user = users[0]
# user = users[3]  # - user with the least # of ratings -> fast model training
print("User is: ", user["_id"])

url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json", encoding="utf-8")
urls = json.load(url_file)

ratings = []
documents = []
for link in user["links"]:
    ratings.append(link["rating"])
    # ? check if the url exists to get the text, otherwise use selenium?
    for url in urls:
        if link["url"] == url["url"]:
            text = url["text"]
            language = url["language"]
            break

    # ? text now has the text of the url
    print(link['url'], " : ", link["rating"])

#     documents.append(preprocess_document(text))
    documents.append(text)
# # Step 2 Pad sequences

# # Convert sentences to sequences of word indices
# sentence_sequences = []
# vocab = ft.get_words()
# print(len(vocab))
# counter = 0
# for sentence in documents:
#     print("new doc ", counter)
#     indices = [ft.get_word_id(word)
#                for word in sentence if word in vocab]
#     sentence_sequences.append(indices)
#     counter = counter + 1

# # print("sentence sequence now")
# # print(sentence_sequences)

# # Pad sequences
# max_sequence_length = max(len(seq) for seq in sentence_sequences)
# padded_sequences = pad_sequences(
#     sentence_sequences, maxlen=max_sequence_length, padding='post', dtype='float32')


# # print("max sequence length: ", max_sequence_length)
# # print("padded sequences")
# # print(padded_sequences)


# # Split data train-test
# documents_train, documents_test, ratings_train, ratings_test = train_test_split(
#     padded_sequences, ratings, test_size=0.2, random_state=42)

# ratings_train = [rating - 1 for rating in ratings_train]
# ratings_test = [rating - 1 for rating in ratings_test]

# ratings_train = np.array(ratings_train)
# ratings_test = np.array(ratings_test)

# embedding_layer = Embedding(
#     input_dim=len(vocab),
#     output_dim=ft.get_dimension(),
#     weights=[ft.get_labels()],
#     trainable=False
# )

# # Create LSTM model
# LSTM_model = Sequential()
# LSTM_model.add(embedding_layer)
# LSTM_model.add(LSTM(128))
# LSTM_model.add(Dense(5, activation='softmax'))


# # Compile LSTM model
# LSTM_model.compile(
#     optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # LSTM_model.compile(
# # optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# print(LSTM_model.summary())
# LSTM_model.fit(documents_train, ratings_train,  validation_data=(
#     documents_test, ratings_test), epochs=10, batch_size=32)


# # Evaluate the model on the test data
# loss, accuracy = LSTM_model.evaluate(
#     documents_test, ratings_test)

# # Print the evaluation results
# print("Test Loss:", loss)
# print("Test Accuracy:", accuracy)
# predictions = LSTM_model.predict(documents_test)
# predicted_classes = np.argmax(predictions, axis=1)
# print(ratings_test)
# print(predicted_classes)

# predictions = LSTM_model.predict(documents_train)
# predicted_classes = np.argmax(predictions, axis=1)
# print(ratings_train)
# print(predicted_classes)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(documents)
sequences = tokenizer.texts_to_sequences(documents)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, padding='post')

# Get the vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # +1 for the reserved padding index

# Create embedding matrix
# fasttext.util.reduce_model(ft, 100)
embedding_dimension = ft.get_dimension()
embedding_matrix = np.zeros((vocab_size, embedding_dimension))
for word, idx in tokenizer.word_index.items():
    embedding_matrix[idx] = ft.get_word_vector(word)

# Create embedding layer
embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dimension,
    weights=[embedding_matrix],
    trainable=False
)

sequence_lengths = [len(seq) for seq in padded_sequences]
max_sequence_length = max(sequence_lengths)

print(vocab_size)
print(max_sequence_length)

# # Create the model
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=(max_sequence_length,)))
# # model.add(tf.keras.layers.Input(shape=(None,)))  # Input layer for sequences
# model.add(embedding_layer)
# model.add(GlobalAveragePooling1D())
# # model.add(Dense(128, activation='relu'))
# # Example layer to flatten the sequence data
# # model.add(tf.keras.layers.Flatten())
# # Add other layers as needed
# model.add(tf.keras.layers.Dense(units=5, activation='softmax'))

# ? LSTM model
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(128))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model = Sequential()
# model.add(embedding_layer)
# # Add a GlobalAveragePooling1D layer to aggregate sequence information
# model.add(GlobalAveragePooling1D())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(5, activation='softmax'))

# Split data train-test
documents_train, documents_test, ratings_train, ratings_test = train_test_split(
    padded_sequences, ratings, test_size=0.2, random_state=42)

ratings_train = [rating - 1 for rating in ratings_train]
ratings_test = [rating - 1 for rating in ratings_test]

ratings_train = np.array(ratings_train)
ratings_test = np.array(ratings_test)

print("Input data shape:", documents_train.shape)
print("Target labels shape:", ratings_train.shape)
print("Input data shape:", documents_test.shape)
print("Target labels shape:", ratings_test.shape)

print(model.summary())

model.fit(documents_train, ratings_train, validation_data=(
    documents_test, ratings_test), epochs=10, batch_size=32)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(
    documents_test, ratings_test)

# Print the evaluation results
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Print the evaluation results
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

predictions = model.predict(documents_test)
predicted_classes = np.argmax(predictions, axis=1)
print(ratings_test)
print(predicted_classes)

predictions = model.predict(documents_train)
predicted_classes = np.argmax(predictions, axis=1)
print(ratings_train)
print(predicted_classes)

number_counts = {}
for num in ratings:
    if num in number_counts:
        number_counts[num] += 1
    else:
        number_counts[num] = 1

# Print the counts
for num, count in number_counts.items():
    print(f"Number {num} appears {count} times")
