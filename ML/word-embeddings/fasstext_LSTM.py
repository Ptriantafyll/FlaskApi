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
from sklearn.utils.class_weight import compute_class_weight

ft = fasttext.load_model(
    r"C:\Users\ptria\source\repos\FlaskApi\ML\word-embeddings\cc.el.300.bin")

user_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\users.json")
users = json.load(user_file)

# Pick a user
user = users[1]

# ? File that contains all the urls in the mongodb cluster
url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls_without_errors.json", encoding="utf-8")
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

# Tokenize documents
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(documents)
# Create sequences of word indices
sequences = tokenizer.texts_to_sequences(documents)

# max_sequence_length = max(len(seq) for seq in sentence_sequences)
# ? use max length 512 instead of the max length of all sequences
max_sequence_length = 512
# Pad sequences
padded_sequences = pad_sequences(
    sequences=sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Train-val
documents_train, documents_val, ratings_train, ratings_val = train_test_split(
    padded_sequences, ratings, test_size=0.2, random_state=42)

# Train-test
documents_train, documents_test, ratings_train, ratings_test = train_test_split(
    padded_sequences, ratings, test_size=0.2, random_state=42)

# Make ratings 0-4 instead of 1-5
ratings_train = [rating - 1 for rating in ratings_train]
ratings_val = [rating - 1 for rating in ratings_val]
ratings_test = [rating - 1 for rating in ratings_test]

# Convert ratings into the required format
ratings_train = np.array(ratings_train)
ratings_val = np.array(ratings_val)
ratings_test = np.array(ratings_test)

class_weights = compute_class_weight('balanced', classes=np.unique(ratings_train), y=ratings_train)
class_weights_dict = dict(enumerate(class_weights))

# Get the vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # +1 for the reserved padding index
print(vocab_size)

# You can reduce dimension from 300 to 100 for faster model
# fasttext.util.reduce_model(ft, 100)
# Create matrix with fasttext embeddings
embedding_dimension = ft.get_dimension()
embedding_matrix = np.zeros((vocab_size, embedding_dimension))
for word, idx in tokenizer.word_index.items():
    embedding_matrix[idx] = ft.get_word_vector(word)

# Input Layer
input_layer = Input((max_sequence_length,))

# Embedding layer
embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dimension,
    weights=[embedding_matrix],
    input_length=max_sequence_length,
    trainable=False)(input_layer)

# LSTM layer
lstm_layer = LSTM(units=512, dropout=0.4)(embedding_layer)

# Output layer
intermediate_layer = Dense(64, activation='relu')(lstm_layer)
output_layer = Dense(1, activation='linear')(intermediate_layer)

# Create lstm model
LSTM_model = Model(inputs=input_layer, outputs=output_layer)
optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
LSTM_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

LSTM_model.summary()  # Print model
# Train model
history = LSTM_model.fit(documents_train, ratings_train,  validation_data=(
    documents_val, ratings_val), epochs=10, batch_size=32, class_weight=class_weights_dict)


# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.savefig(r"C:\Users\ptria\source\repos\FlaskApi\images\fasttext\mse.png")
plt.legend()
plt.show()

# Plot training and validation MAE
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.savefig(r"C:\Users\ptria\source\repos\FlaskApi\images\fasttext\mae.png")
plt.show()


predictions = LSTM_model.predict(documents_val)
predicted_ratings = [max(0, min(round(x),4)) for x in predictions.flatten()]
# Compute the confusion matrix
cm = confusion_matrix(ratings_val, predicted_ratings)
# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(r"C:\Users\ptria\source\repos\FlaskApi\images\fasttext\val_cm.png")
plt.show()

predictions = LSTM_model.predict(documents_train)
predicted_ratings = [max(0, min(round(x),4)) for x in predictions.flatten()]

cm = confusion_matrix(ratings_train, predicted_ratings)
# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(r"C:\Users\ptria\source\repos\FlaskApi\images\fasttext\train_cm.png")
plt.show()

predictions = LSTM_model.predict(documents_test)
predicted_ratings = [max(0, min(round(x),4)) for x in predictions.flatten()]
# Compute the confusion matrix
cm = confusion_matrix(ratings_test, predicted_ratings)
# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(r"C:\Users\ptria\source\repos\FlaskApi\images\fasttext\test_cm.png")
plt.show()
