import json
import mongoDB_connection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import spacy
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


mongoDB_connection.connect_to_mongodb()
db = mongoDB_connection.db
users = db['user'].find()

# print(users[0]["_id"])
user = users[0]

# todo: Step 2: Load and preprocess labeled urls of the user
f = open(r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json",
         encoding="utf-8")
urls = json.load(f)
greek_stop_words_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\greek_stop_words.json", encoding="utf8")
greek_stop_words = json.load(greek_stop_words_file)

# for link in user["links"]:
# print(link)
print(user["links"][0]["url"])
print(urls[0]["url"])

nlp_greek = spacy.load("el_core_news_sm")
nlp_english = spacy.load("en_core_web_sm")

for link in user["links"]:
    # ? check if the url exists to get the text, otherwise use selenium?
    for url in urls:
        if link["url"] == url["url"]:
            text = url["text"]
            language = url["language"]
            break

    # ? text now has the text of the url
    tokenized_documents = []
    print(link['url'], " : ", language)

    lower_case_text = text.lower()  # ? make text lowercase
    tokenizer = RegexpTokenizer(r'\w+')  # ? tokenize and remove punctuation
    words = tokenizer.tokenize(lower_case_text)
    words = [word for word in words if word.isalpha()]  # ? remove numbers

    # print(words)

    # ? remove english and greek stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [
        w for w in words if not w in stop_words and not w in greek_stop_words and len(w) > 1]
    words.append(filtered_words)

    if 'en' in language:
        # print("stemming english")
        ps = PorterStemmer()
        stemmed_words = [ps.stem(word) for word in filtered_words]
        tokenized_documents.append(stemmed_words)
        # print(stemmed_words)

    if 'el' in language:
        # print("lemmatizing greek")
        lemmatized_words = []
        lemmatized_words = [nlp_english(token)[0].lemma_ if token.isascii(
        ) else nlp_greek(token)[0].lemma_ for token in filtered_words]
        tokenized_documents.append(lemmatized_words)
        # print(lemmatized_words)


# for doc in tokenized_documents:
#     print(doc)


# todo: Step 3: TF-IDF Transformation (of the text of the urls)
# Load the vectorizer from the file
# with open(r"C:\Users\ptria\source\repos\FlaskApi\ML\tf-idf\tfidf_vectorizer.pkl", 'rb') as f:
#     loaded_tfidf_vectorizer = pickle.load(f)

documents = [" ".join(tokens) for tokens in tokenized_documents]
# new_text_tfidf = loaded_tfidf_vectorizer.transform(new_text)

# todo: Step 4: Model Training -- train test split, logistic regression.fit
# todo: Step 5: Model Evaluation -- logreg.predict, accuracy score

# ?logistic regression


# ?SVM classifier


# ? decision tree (για σύγκριση)
