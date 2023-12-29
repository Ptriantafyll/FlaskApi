import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, LSTM, Dense, Dropout, Input
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from gensim.models import Word2Vec
import json
import numpy as np
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# This is needed to import a function from different directory
import sys
sys.path.append(r"C:\Users\ptria\source\repos\FlaskApi\ML")
from functions import normalize
import matrix_factorization

# ? load word2vec model from file
word2vec_model = Word2Vec.load(
    r"C:\Users\ptria\source\repos\FlaskApi\ML\word-embeddings\word2vec_model")

# Returns user-url matrix as pandas DataFrame 
df = matrix_factorization.perform_martix_factorization()

# ? File that contains all the urls in the mongodb cluster
url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls_without_errors.json", encoding="utf-8")
urls = json.load(url_file)

# ? The following lines are needed only the first time you run the code
# import nltk
# nltk.download('stopwords')

# pick a user from pandas df
user = df.index[1]

ratings = []
documents = []
# take english stopwords from nltk
stop_words = set(stopwords.words('english'))
# ? File that contains the greek stopwords
greek_stop_words_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\greek_stop_words.json", encoding="utf8")
greek_stop_words = json.load(greek_stop_words_file)
# todo: change iterations and get data from pandas df
counter = 0
for url in df.columns:
    ratings.append(df.loc[user,url])
    counter = counter + 1

    # ? check if the url exists to get the text
    for link in urls:
        if url == link["url"]:
            text = link["text"]
            url_language = link["language"]
            break

    # ? text now has the text of the url
    lower_case_text = normalize.strip_accents_and_lowercase(text)
    tokenizer = RegexpTokenizer(r'\w+')  # ? tokenize and remove punctuation
    words = tokenizer.tokenize(lower_case_text)
    words = [word for word in words if word.isalpha(
    ) and word not in stop_words and word not in greek_stop_words]  # ? remove numbers and stopwords
    documents.append(words)


# Convert sentences to sequences of word indices
sentence_sequences = []
for sentence in documents:
    # todo: if word does not exist in vocab -> skip maybe?
    indices = [word2vec_model.wv.key_to_index[word]
               for word in sentence if word in word2vec_model.wv]
    sentence_sequences.append(indices)

# Pad sequences
# max_sequence_length = max(len(seq) for seq in sentence_sequences)
# ? use max length 512 instead of the max length of all sequences
max_sequence_length = 512
print("max sequence length: ", max_sequence_length)

# ? Word2Vec model embeddings
embeddings = pad_sequences(
    sentence_sequences, maxlen=max_sequence_length, padding='post', truncating='post')

# Get the vocabulary as a set
vocabulary = list(word2vec_model.wv.index_to_key)

# Split data into train-test
documents_train, documents_test, ratings_train, ratings_test = train_test_split(
    embeddings, ratings, test_size=0.2, random_state=42)

# Make ratings 0-4 instead of 1-5
ratings_train = [rating - 1 for rating in ratings_train]
ratings_test = [rating - 1 for rating in ratings_test]

# Convert ratings into the required format
ratings_train = np.array(ratings_train)
ratings_test = np.array(ratings_test)

# Input layer
input_layer = Input((max_sequence_length,), name="input layer")

# Embedding layer
embedding_layer = Embedding(input_dim=word2vec_model.wv.vectors.shape[0],
                            output_dim=word2vec_model.wv.vectors.shape[1],
                            weights=[word2vec_model.wv.vectors],
                            input_length=max_sequence_length,
                            trainable=False, name="word2vec_embeddings")(input_layer)

# LSTM layer
lstm_layer = LSTM(units=512, dropout=0.2)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)

# Output layer
intermediate_layer = Dense(64, activation='relu')(dropout_layer)
dropout_layer = Dropout(0.2)(intermediate_layer)
output_layer = Dense(5, activation='softmax')(dropout_layer)

# Create lstm model
LSTM_model = Model(inputs=input_layer, outputs=output_layer)
LSTM_model.compile(loss='sparse_categorical_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

LSTM_model.summary()  # Print model
# Train model
LSTM_model.fit(documents_train, ratings_train,  validation_data=(
    documents_test, ratings_test), epochs=20, batch_size=32)


predictions = LSTM_model.predict(documents_test)
predicted_classes = np.argmax(predictions, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(ratings_test, predicted_classes)
# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

predictions = LSTM_model.predict(documents_train)
predicted_classes = np.argmax(predictions, axis=1)

cm = confusion_matrix(ratings_train, predicted_classes)
# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
