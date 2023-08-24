from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from gensim.models import Word2Vec
import gensim
import json
from gensim_word_embeddings import preprocess_document
import numpy as np

word2vec_model = Word2Vec.load(
    r"C:\Users\ptria\source\repos\FlaskApi\ML\word-embeddings\word2vec_model")

# Step 1 Preprocess user's ratings
user_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\users.json")
users = json.load(user_file)

# todo ? cluster of users
user = users[11]
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

    documents.append(preprocess_document(text))
# Step 2 Pad sequences

# Convert sentences to sequences of word indices
sentence_sequences = []
for sentence in documents:
    indices = [word2vec_model.wv.key_to_index[word]
               for word in sentence if word in word2vec_model.wv]
    sentence_sequences.append(indices)

# print("sentence sequence now")
# print(sentence_sequences)

# Pad sequences
max_sequence_length = max(len(seq) for seq in sentence_sequences)
padded_sequences = pad_sequences(
    sentence_sequences, maxlen=max_sequence_length, padding='post', dtype='float32')


# print("max sequence length: ", max_sequence_length)
# print("padded sequences")
# print(padded_sequences)


# Split data train-test
documents_train, documents_test, ratings_train, ratings_test = train_test_split(
    padded_sequences, ratings, test_size=0.2, random_state=42)

ratings_train = [rating - 1 for rating in ratings_train]
ratings_test = [rating - 1 for rating in ratings_test]

ratings_train = np.array(ratings_train)
ratings_test = np.array(ratings_test)

# Create embedding layer
embedding_layer = Embedding(
    input_dim=word2vec_model.wv.vectors.shape[0],
    output_dim=word2vec_model.wv.vectors.shape[1],
    weights=[word2vec_model.wv.vectors],
    trainable=False
)

# Create LSTM model
LSTM_model = Sequential()
LSTM_model.add(embedding_layer)
LSTM_model.add(LSTM(128))
LSTM_model.add(Dense(5, activation='softmax'))


# Compile LSTM model
LSTM_model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


LSTM_model.fit(documents_train, ratings_train,  validation_data=(
    documents_test, ratings_test), epochs=10, batch_size=32)


# Step 4: Model Evaluation
ratings_pred = LSTM_model.predict(documents_test)
accuracy_lstm = accuracy_score(ratings_test, ratings_pred)
print("Decision Tree Accuracy:", accuracy_lstm)
# Calculate Mean Squared Error
mse = mean_squared_error(ratings_test, ratings)
print("Mean Squared Error:", mse)
# Calculate F1 score
f1 = f1_score(ratings_test, ratings_pred, average="weighted")
print("F1 Score:", f1)
# Calculate R2 score
r2 = r2_score(ratings_test, ratings_pred)
print("R2 Score:", r2)

print(ratings_pred)
print(ratings_test)
print(ratings_train)
