
# ? data preprocessing in this file

import json
import spacy
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


# user_file = open("users.json")
# users = json.load(user_file)

url_file = open("urls.json")
urls = json.load(url_file)


# for user in users:
# for link in user["links"]:
# print(link['url'])
# print(link["rating"])s

# for link in urls:
# if(link['language'] != ""):
# continue

# print(link['url'], " : ", link['language'])


# nltk.download('stopwords')

# print(urls[0]["url"])
# print(urls[0]['text'])
text = urls[0]['text']

lower_case_text = text.lower()  # ? make words lowercase
tokenizer = RegexpTokenizer(r'\w+')  # ? tokenizing and remove puunctuation
words = tokenizer.tokenize(lower_case_text)
# words = [word for word in words if word.isalpha()]  # ? remove numbers
# print(words)


# nltk.download('stopwords')


if 'en' in urls[0]['language']:
    # remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if not w in stop_words]
    # print(filtered_words)

    print("stemming")
    ps = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(ps.stem(word))

    # print(stemmed_words)


if 'el' in urls[0]['language']:
    # ? lemmatizing in greek
    # Load the Greek language model in spacy
    nlp = spacy.load("el_core_news_sm")

    # remove stop words
    greek_stop_words = set(stopwords.words('greek'))
    filtered_words = [w for w in words if not w in greek_stop_words]

    # lemmatize in greek
    lemmatized_words = []
    for word in filtered_words:
        lemmatized_words.append(nlp(word)[0].lemma_)
    print(lemmatized_words)


# todo: create tf-idf vectors tfidfvectorizer from sklearn


# todo: documents = all website texts
# documents = [" ".join(tokens) for tokens in tokenized_documents]

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_vectors = tfidf_vectorizer.fit_transform(documents)
# tfidf_vectors = tfidf_vectors.toarray()
