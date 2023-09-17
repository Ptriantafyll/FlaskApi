import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, LSTM, Dense, Input
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from gensim.models import Word2Vec
import json
import numpy as np
import unicodedata
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

word2vec_model = Word2Vec.load(
    r"C:\Users\ptria\source\repos\FlaskApi\ML\word-embeddings\word2vec_model")

# Step 1 Preprocess user's ratings
user_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\users.json")
users = json.load(user_file)


def strip_accents_and_lowercase(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn').lower()


# todo ? cluster of users
user = users[32]
# user = users[3] #- user with the least # of ratings -> fast model training
print("User is: ", user["_id"])

url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json", encoding="utf-8")
urls = json.load(url_file)

ratings = []
documents = []
raw_documents = []
stop_words = set(stopwords.words('english'))
greek_stop_words_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\greek_stop_words.json", encoding="utf8")
greek_stop_words = json.load(greek_stop_words_file)
for link in user["links"]:
    ratings.append(link["rating"])
    # ? check if the url exists to get the text, otherwise use selenium?
    for url in urls:
        if link["url"] == url["url"]:
            text = url["text"]
            language = url["language"]
            break

    # ? text now has the text of the url
    # print(link['url'], " : ", link["rating"])
    lower_case_text = strip_accents_and_lowercase(text)
    tokenizer = RegexpTokenizer(r'\w+')  # ? tokenize and remove punctuation
    words = tokenizer.tokenize(lower_case_text)
    words = [word for word in words if word.isalpha(
    ) and word not in stop_words and word not in greek_stop_words]  # ? remove numbers and stopwords
    raw_documents.append(text)
    documents.append(words)

# print('unprocessed document: ', raw_documents[0])
print('preprocessed document: ', documents[0])
word = "data"
print(word2vec_model.wv[word].shape)
# Step 2 Pad sequences

# Convert sentences to sequences of word indices
sentence_sequences = []
for sentence in documents:
    # todo: if word does not exist in vocab -> skip maybe?
    indices = [word2vec_model.wv.key_to_index[word]
               for word in sentence if word in word2vec_model.wv]
    sentence_sequences.append(indices)

print("sentence sequence now")
print(sentence_sequences[0])

w = "κιθαρα"
print("most similar to: ", w)
print(word2vec_model.wv.most_similar(positive=w, topn=20))

w = "καθαρος"
print("most similar to: ", w)
print(word2vec_model.wv.most_similar(positive=w, topn=20))

w = "children"
print("most similar to: ", w)
print(word2vec_model.wv.most_similar(positive=w, topn=20))

# Pad sequences
# max_sequence_length = max(len(seq) for seq in sentence_sequences)
max_sequence_length = 512  # ? use max length 512
print("max sequence length: ", max_sequence_length)
embeddings = pad_sequences(
    sentence_sequences, maxlen=max_sequence_length, padding='post', truncating='post')


print("padded sequences")
print(embeddings.shape)
print(type(embeddings))
print(embeddings[0])
# print(embeddings[1])

# Get the vocabulary as a set
vocabulary = list(word2vec_model.wv.index_to_key)
print(len(vocabulary))
print(list(vocabulary)[:10])
print(vocabulary[len(vocabulary)-1])

print(word2vec_model.wv.get_index("τρυπα"))
print(word2vec_model.wv.key_to_index["τρυπα"])


# Split data train-test
documents_train, documents_test, ratings_train, ratings_test = train_test_split(
    embeddings, ratings, test_size=0.2, random_state=42)

ratings_train = [rating - 1 for rating in ratings_train]
ratings_test = [rating - 1 for rating in ratings_test]

ratings_train = np.array(ratings_train)
ratings_test = np.array(ratings_test)

print(word2vec_model.wv.vectors.shape)
print(embeddings.shape)
print(embeddings.shape[0])
print(embeddings.shape[1])

# Input layer
x_in = Input((max_sequence_length,))

# Embedding layer
x = Embedding(input_dim=word2vec_model.wv.vectors.shape[0],
              output_dim=word2vec_model.wv.vectors.shape[1],
              weights=[word2vec_model.wv.vectors],
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
