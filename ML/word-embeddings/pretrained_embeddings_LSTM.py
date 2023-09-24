import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import Model
import json
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Input
from sklearn.model_selection import train_test_split
import fasttext
import fasttext.util
import numpy as np
import tensorflow as tf

ft = fasttext.load_model(
    r"C:\Users\ptria\source\repos\FlaskApi\ML\word-embeddings\cc.el.300.bin")

user_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\users.json")
users = json.load(user_file)

# # todo ? cluster of users
user = users[1]
# user = users[3]  # - user with the least # of ratings -> fast model training
print("User is: ", user["_id"])

url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json", encoding="utf-8")
urls = json.load(url_file)

# Preprocess user's ratings
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
    documents.append(text)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(documents)
sequences = tokenizer.texts_to_sequences(documents)

# max_sequence_length = max(len(seq) for seq in sentence_sequences)
max_sequence_length = 512
padded_sequences = pad_sequences(
    sequences=sequences, maxlen=max_sequence_length, padding='post', truncating='post')


# Split data train-test
documents_train, documents_test, ratings_train, ratings_test = train_test_split(
    padded_sequences, ratings, test_size=0.2, random_state=42)

ratings_train = [rating - 1 for rating in ratings_train]
ratings_test = [rating - 1 for rating in ratings_test]

ratings_train = np.array(ratings_train)
ratings_test = np.array(ratings_test)

# Get the vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # +1 for the reserved padding index
print(vocab_size)

# Create embedding matrix
# fasttext.util.reduce_model(ft, 100)
embedding_dimension = ft.get_dimension()
embedding_matrix = np.zeros((vocab_size, embedding_dimension))
for word, idx in tokenizer.word_index.items():
    embedding_matrix[idx] = ft.get_word_vector(word)

# Input Layer
x_in = Input((max_sequence_length,))

# Embedding layer
x = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dimension,
    weights=[embedding_matrix],
    input_length=max_sequence_length,
    trainable=False)(x_in)

# LSTM layer
# x = LSTM(units=512, dropout=0.2, return_sequences=True)(x)
x = LSTM(units=512, dropout=0.2)(x)

# Output layer
x = Dense(64, activation='relu')(x)
y_out = Dense(5, activation='softmax')(x)

LSTM_model = Model(inputs=x_in, outputs=y_out)
LSTM_model.compile(loss='sparse_categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

LSTM_model.summary()
print(documents_train.shape)
print(documents_test.shape)
print(ratings_train.shape)
print(ratings_test.shape)
LSTM_model.fit(documents_train, ratings_train,  validation_data=(
    documents_test, ratings_test), epochs=10, batch_size=32)

# Evaluate the model on the test data
loss, accuracy = LSTM_model.evaluate(
    documents_test, ratings_test)

# Print the evaluation results
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

predictions = LSTM_model.predict(documents_test)
predicted_classes = np.argmax(predictions, axis=1)
print(ratings_test)
print(predicted_classes)

# Compute the confusion matrix
cm = confusion_matrix(ratings_test, predicted_classes)
print(cm)

# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

predictions = LSTM_model.predict(documents_train)
predicted_classes = np.argmax(predictions, axis=1)
print(ratings_train)
print(predicted_classes)

cm = confusion_matrix(ratings_train, predicted_classes)
print(cm)

# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()