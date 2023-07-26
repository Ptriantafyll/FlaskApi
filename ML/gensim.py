
# ? here will be the code for gensim word embeddings
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize


def preprocess_document(doc):
    tokens = word_tokenize(doc.lower())  # ?Tokenize and lowercase
    # ?Remove punctuation
    filtered_tokens = [token for token in tokens if token.isalpha()]
    return filtered_tokens


# ? gensim word embeddings
# todo: Take docs from mongo db
docs = [
    "apple pear banana",
    "apple mango"
]
documents = [preprocess_document(doc) for doc in docs]
print(documents)


# Train the Word2Vec model
model = Word2Vec(sentences=documents, vector_size=100,
                 window=5, min_count=1, workers=4)

# Save the model to a file (optional)
# model.save("word2vec_model")

# Load the model from a file (if needed)
# model = Word2Vec.load("word2vec_model")
