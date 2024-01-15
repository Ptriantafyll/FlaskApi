
# ? here will be the code for gensim word embeddings
import json
import logging
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from functions import normalize


def preprocess_document(doc):
    stop_words = set(stopwords.words('english'))
    greek_stop_words_file = open(
        r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\greek_stop_words.json", encoding="utf8")
    greek_stop_words = json.load(greek_stop_words_file)

    tokenizer = RegexpTokenizer(r'\w+')  # ? tokenizes and removes puunctuation
    lower_case_text = normalize.strip_accents_and_lowercase(doc)

    tokens = tokenizer.tokenize(lower_case_text)

    # ? Remove punctuation and stopwords
    filtered_tokens = [token for token in tokens if token.isalpha(
    ) and token not in stop_words and token not in greek_stop_words]

    return filtered_tokens


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# ? File that contains all the urls in the mongodb cluster
url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json", encoding="utf8")
urls = json.load(url_file)

documents = [preprocess_document(url['text']) for url in urls]

# Train the Word2Vec model
model = Word2Vec(sentences=documents, vector_size=100,
                 window=10, min_count=1, workers=4, sg=1)
model.train(documents, total_examples=len(documents), epochs=10)


# Save the model to a file (optional)
# model.save(r"C:\Users\ptria\source\repos\FlaskApi\ML\word-embeddings\word2vec_model")

# ? example most similar words
# w = "μπύρα"
# print("most similar to: ", w)
# print(model.wv.most_similar(positive=w, topn=10))
