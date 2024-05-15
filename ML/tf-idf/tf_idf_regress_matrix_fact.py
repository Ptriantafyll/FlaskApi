import json
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                             r2_score, mean_absolute_error)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# This is needed to import a function from different directory
import sys
sys.path.append(r"C:\Users\ptria\source\repos\FlaskApi\ML")
from functions import normalize
import matrix_factorization

# ? you need the following 3 lines only the first time you run this file
# import nltk
# nltk.download("stopwords")
# nltk.download("punkt")

df = matrix_factorization.perform_martix_factorization()

# ? File that contains all the urls in the mongodb cluster
url_file = open(r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json",
                encoding="utf-8")
urls = json.load(url_file)

# pick a user from pandas df
user = df.index[1]

# ? take greek stop words from file
greek_stop_words_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\greek_stop_words.json", encoding="utf8")
greek_stop_words = json.load(greek_stop_words_file)
# take english stopwords from nltk
stop_words = set(stopwords.words('english'))

nlp_greek = spacy.load("el_core_news_sm")
nlp_english = spacy.load("en_core_web_sm")

# ? tokenize text of rated urls
ratings = []
tokenized_documents = []
not_found =[]
counter = 0
for url in df.columns:
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
    words = [word for word in words if word.isalpha()]  # ? remove numbers

    # ? remove english and greek stop words
    filtered_words = [
        w for w in words if not w in stop_words and not w in greek_stop_words and len(w) > 1]
    words.append(filtered_words)

    # ? If the document is in english use stemming
    if 'en' in url_language:
        ps = PorterStemmer()
        stemmed_words = [ps.stem(word) for word in filtered_words]
        tokenized_documents.append(stemmed_words)
        ratings.append(df.loc[user,url])
    # ? If the document is in greek use lemmatizing
    elif 'el' in url_language:
        lemmatized_words = []
        lemmatized_words = [nlp_english(token)[0].lemma_ if token.isascii(
        ) else nlp_greek(token)[0].lemma_ for token in filtered_words]
        tokenized_documents.append(lemmatized_words)
        ratings.append(df.loc[user,url])
    else:
        not_found.append(url)


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
logreg_model = LogisticRegression(max_iter=10000)
logreg_model.fit(documents_train, ratings_train)

# LogReg Model Evaluation
ratings_pred = logreg_model.predict(documents_test)
accuracy = accuracy_score(ratings_test, ratings_pred)
print(ratings_test, ratings_pred)
print("Accuracy:", accuracy)
# Calculate Mean Squared Error
mse = mean_squared_error(ratings_test, ratings_pred)
print("Mean Squared Error:", mse)
mae = mean_absolute_error(ratings_test, ratings_pred)
print("Mean Absolute Error:", mae)
# Calculate F1 score
f1 = f1_score(ratings_test, ratings_pred, average="weighted")
print("F1 Score:", f1)
# Calculate R2 score
r2 = r2_score(ratings_test, ratings_pred)
print("R2 Score:", r2)

# Compute the confusion matrix
cm = confusion_matrix(ratings_test, ratings_pred)
# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Logistic Regression Confusion Matrix')
plt.show()


# SVM Model Training
svm_model = SVC(kernel='rbf')
# svm_model = SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
#                    decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
#                    max_iter=-1, probability=False, random_state=None, shrinking=True,
#                    tol=0.001, verbose=False)
svm_model.fit(documents_train, ratings_train)

# SVM Model Evaluation
ratings_pred_svm = svm_model.predict(documents_test)
print(ratings_test, ratings_pred_svm)
accuracy_svm = accuracy_score(ratings_test, ratings_pred_svm)
print("SVM Accuracy:", accuracy_svm)
# Calculate Mean Squared Error
mse = mean_squared_error(ratings_test, ratings_pred_svm)
print("Mean Squared Error:", mse)
mae = mean_absolute_error(ratings_test, ratings_pred_svm)
print("Mean Absolute Error:", mae)
# Calculate F1 score
f1 = f1_score(ratings_test, ratings_pred_svm, average="weighted")
print("F1 Score:", f1)
# Calculate R2 score
r2 = r2_score(ratings_test, ratings_pred_svm)
print("R2 Score:", r2)

# Compute the confusion matrix
cm = confusion_matrix(ratings_test, ratings_pred_svm)
# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('SVM Confusion Matrix')
plt.show()

# ? decision tree (για σύγκριση)

# Decision Tree Model Training
tree_model = DecisionTreeClassifier()
tree_model.fit(documents_train, ratings_train)

# Decision Tree Model Evaluation
ratings_pred_tree = tree_model.predict(documents_test)
print(ratings_test, ratings_pred_tree)
accuracy_tree = accuracy_score(ratings_test, ratings_pred_tree)
print("Decision Tree Accuracy:", accuracy_tree)
# Calculate Mean Squared Error
mse = mean_squared_error(ratings_test, ratings_pred_tree)
print("Mean Squared Error:", mse)
mae = mean_absolute_error(ratings_test, ratings_pred_tree)
print("Mean Absolute Error:", mae)
# Calculate F1 score
f1 = f1_score(ratings_test, ratings_pred_tree, average="weighted")
print("F1 Score:", f1)
# Calculate R2 score
r2 = r2_score(ratings_test, ratings_pred_tree)
print("R2 Score:", r2)

# Compute the confusion matrix
cm = confusion_matrix(ratings_test, ratings_pred_tree)
# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Decision Tree Confusion Matrix')
plt.show()


