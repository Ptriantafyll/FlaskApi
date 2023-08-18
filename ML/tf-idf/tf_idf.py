import json
import spacy
import nltk
import pickle
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json", encoding="utf8")

urls = json.load(url_file)

# nltk.download('stopwords')

nlp_greek = spacy.load("el_core_news_sm")
nlp_english = spacy.load("en_core_web_sm")

greek_stop_words_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\greek_stop_words.json", encoding="utf8")
greek_stop_words = json.load(greek_stop_words_file)

print(len(urls))

tokenized_documents = []
for i in range(len(urls)):
    text = urls[i]['text']
    print(urls[i]['url'], " : ", urls[i]['language'])

    lower_case_text = text.lower()  # ? make text lowercase
    tokenizer = RegexpTokenizer(r'\w+')  # ? tokenize and remove puunctuation
    words = tokenizer.tokenize(lower_case_text)
    words = [word for word in words if word.isalpha()]  # ? remove numbers

    # print(words)

    # ? remove english and greek stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [
        w for w in words if not w in stop_words and not w in greek_stop_words and len(w) > 1]
    words.append(filtered_words)

    if 'en' in urls[i]['language']:
        # print("stemming english")
        ps = PorterStemmer()
        stemmed_words = [ps.stem(word) for word in filtered_words]
        tokenized_documents.append(stemmed_words)
        # print(stemmed_words)

    if 'el' in urls[i]['language']:
        # print("lemmatizing greek")
        lemmatized_words = []
        lemmatized_words = [nlp_english(token)[0].lemma_ if token.isascii(
        ) else nlp_greek(token)[0].lemma_ for token in filtered_words]
        tokenized_documents.append(lemmatized_words)
        # print(lemmatized_words)

# for doc in tokenized_documents:
#   print(doc)


# CREATE TF IDF VECTORIZER
documents = [" ".join(tokens) for tokens in tokenized_documents]

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(documents)
tfidf_vectors = tfidf_vectors.toarray()
feature_names = tfidf_vectorizer.get_feature_names_out()

# for vector in tfidf_vectors:
#   print(vector)

# ? print nonzero values
non_zero_indices = np.nonzero(tfidf_vectors)
for row, col in zip(*non_zero_indices):
    print(
        f"Document {row}, Term '{feature_names[col]}': {tfidf_vectors[row, col]}")


# Save the vectorizer to a file
with open(r"C:\Users\ptria\source\repos\FlaskApi\ML\tfidf_vectorizer.pkl", 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
