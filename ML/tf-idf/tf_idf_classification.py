from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import r2_score, f1_score, mean_squared_error
import json
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import spacy
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from functions import normalize

# ? you need the following 3 lines only the first time you run this file
# import nltk
# nltk.download("stopwords")
# nltk.download("punkt")
global nlp_greek
global nlp_english
nlp_greek = spacy.load("el_core_news_sm")
nlp_english = spacy.load("en_core_web_sm")


def text_to_words(text):
    lower_case_text = normalize.strip_accents_and_lowercase(text)
    tokenizer = RegexpTokenizer(r'\w+')  # ? tokenize and remove punctuation
    words = tokenizer.tokenize(lower_case_text)
    words = [word for word in words if word.isalpha()]  # ? remove numbers

    # ? remove english and greek stop words
    filtered_words = [
        w for w in words if not w in stop_words and not w in greek_stop_words and len(w) > 1]
    # words.append(filtered_words)

    return filtered_words


def perform_stemming(filtered_words):
    ps = PorterStemmer()
    stemmed_words = [ps.stem(word) for word in filtered_words]

    return stemmed_words


def perform_lemmatization(filtered_words):
    
    lemmatized_words = []
    lemmatized_words = [nlp_english(token)[0].lemma_ if token.isascii(
    ) else nlp_greek(token)[0].lemma_ for token in filtered_words]

    return lemmatized_words


def evaluate_model(model):
    ratings_pred = model.predict(documents_test)
    accuracy = accuracy_score(ratings_test, ratings_pred)
    print("Accuracy:", accuracy)
    # Calculate Mean Squared Error
    mse = mean_squared_error(ratings_test, ratings_pred)
    print("Mean Squared Error:", mse)
    # Calculate F1 score
    f1 = f1_score(ratings_test, ratings_pred, average="weighted")
    print("F1 Score:", f1)
    # Calculate R2 score
    r2 = r2_score(ratings_test, ratings_pred)
    print("R2 Score:", r2)


# ? File that contains all the users in the mongodb cluster
user_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\users.json", encoding="utf8")
users = json.load(user_file)

# ? File that contains all the urls in the mongodb cluster
url_file = open(r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json",
                encoding="utf-8")
urls = json.load(url_file)

# ? Pick a user
user = users[11]

# ? take greek stop words from file
greek_stop_words_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\greek_stop_words.json", encoding="utf8")
greek_stop_words = json.load(greek_stop_words_file)
# take english stopwords from nltk
stop_words = set(stopwords.words('english'))

# ? tokenize text of rated urls
ratings = []
tokenized_documents = []
for link in user["links"]:
    ratings.append(link["rating"])
    # ? check if the url exists to get the text
    for url in urls:
        if link["url"] == url["url"]:
            text = url["text"]
            break
    # ? text now has the text of the url

    filtered_words = text_to_words(text)
    # ? If the document is in english use stemming
    if 'en' in url["language"]:
        stemmed_words = perform_stemming(filtered_words)
        tokenized_documents.append(stemmed_words)

    # ? If the document is in greek use lemmatizing
    if 'el' in url["language"]:
        lemmatized_words = perform_lemmatization(filtered_words)
        tokenized_documents.append(lemmatized_words)

# TF-IDF Transformation
# Load the vectorizer from the file
with open(r"C:\Users\ptria\source\repos\FlaskApi\ML\tf-idf\tfidf_vectorizer.pkl", 'rb') as f:
    loaded_tfidf_vectorizer = pickle.load(f)

documents = [" ".join(tokens) for tokens in tokenized_documents]
documents_tfidf = loaded_tfidf_vectorizer.transform(documents)

# Split data into training-testing
documents_train, documents_test, ratings_train, ratings_test = train_test_split(
    documents_tfidf, ratings, test_size=0.2, random_state=21)

# Logistic Regression Model Training
logreg_model = LogisticRegression(max_iter=2000)
logreg_model.fit(documents_train, ratings_train)

# LogReg Model Evaluation
print("Evaluating logistic regression:")
evaluate_model(logreg_model)
print("Finished evaluating logistic regression \n")

# SVM Model Training
svm_model = SVC(kernel='rbf')
# svm_model = SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#                    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
#                    max_iter=-1, probability=False, random_state=None, shrinking=True,
#                    tol=0.001, verbose=False)
svm_model.fit(documents_train, ratings_train)

# SVM Model Evaluation
print("Evaluating logistic regression:")
evaluate_model(svm_model)
print("Finished evaluating logistic regression\n")


# ? decision tree (για σύγκριση)
# Decision Tree Model Training
tree_model = DecisionTreeClassifier()
tree_model.fit(documents_train, ratings_train)

# Decision Tree Model Evaluation
evaluate_model(tree_model)