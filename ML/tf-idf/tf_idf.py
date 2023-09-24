import json
import spacy
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from functions import normalize

# ? File that contains all the urls in the mongodb cluster
url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json", encoding="utf8")
urls = json.load(url_file)

# ? The following lines are needed only the first time you run the code
# import nltk
# nltk.download('stopwords')

nlp_greek = spacy.load("el_core_news_sm")
nlp_english = spacy.load("en_core_web_sm")

# ? File that contains the greek stopwords
greek_stop_words_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\greek_stop_words.json", encoding="utf8")
greek_stop_words = json.load(greek_stop_words_file)

# ? Bag of words tokenizing
tokenized_documents = []
for i in range(len(urls)):
    text = urls[i]['text']

    lower_case_text = normalize.strip_accents_and_lowercase(text)
    tokenizer = RegexpTokenizer(r'\w+')  # ? tokenize and remove punctuation
    words = tokenizer.tokenize(lower_case_text)
    words = [word for word in words if word.isalpha()]  # ? remove numbers

    # ? remove english and greek stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [
        w for w in words if not w in stop_words and not w in greek_stop_words and len(w) > 1]
    words.append(filtered_words)

    # ? If the document is in english use stemming
    if 'en' in urls[i]['language']:
        ps = PorterStemmer()
        stemmed_words = [ps.stem(word) for word in filtered_words]
        tokenized_documents.append(stemmed_words)

    # ? If the document is in greek use lemmatizing
    if 'el' in urls[i]['language']:
        lemmatized_words = []
        lemmatized_words = [nlp_english(token)[0].lemma_ if token.isascii(
        ) else nlp_greek(token)[0].lemma_ for token in filtered_words]
        tokenized_documents.append(lemmatized_words)


documents = [" ".join(tokens) for tokens in tokenized_documents]

# ? Create tf-idf vectorizer and fit on documents
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(documents)
tfidf_vectors = tfidf_vectors.toarray()


# ? print nonzero values
# feature_names = tfidf_vectorizer.get_feature_names_out()
# non_zero_indices = np.nonzero(tfidf_vectors)
# for row, col in zip(*non_zero_indices):
#     print(
#         f"Document {row}, Term '{feature_names[col]}': {tfidf_vectors[row, col]}")


# Save the vectorizer to a file
with open(r"C:\Users\ptria\source\repos\FlaskApi\ML\tf-idf\tfidf_vectorizer.pkl", 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
