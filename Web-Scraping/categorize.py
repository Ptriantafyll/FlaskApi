from gensim import corpora
from gensim.models import LdaModel
import json
import mongoDB_connection
import nltk
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import unicodedata


def strip_accents_and_lowercase(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()

# Connect to MongoDB and retrieve text data from the collection
# mongoDB_connection.connect_to_mongodb()
# db = mongoDB_connection.db
# db.command("collMod", "url", validator=mongoDB_connection.validator())


url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json", encoding="utf8")
urls = json.load(url_file)

greek_stop_words_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\greek_stop_words.json", encoding="utf8")
greek_stop_words = json.load(greek_stop_words_file)


# nltk.download('stopwords')
print(len(urls))

documents = []
for i in range(len(urls)):
    text = urls[i]['text']
    print(urls[i]['url'], " : ", urls[i]['language'])

    text = strip_accents_and_lowercase(urls[i]['text'])
    tokenizer = RegexpTokenizer(r'\w+')  # ? tokenize and remove puunctuation
    words = tokenizer.tokenize(text)
    words = [word for word in words if word.isalpha()]  # ? remove numbers

    # remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_words = [
        w for w in words if not w in stop_words and not w in greek_stop_words and len(w) > 1]
    documents.append(filtered_words)

# for doc in documents:
    # print(doc)

# Create a dictionary and a corpus
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]

# Train the LDA model
num_topics = 5
lda = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

# Print the topics
for i in range(num_topics):
    print(f"Topic {i}: {lda.print_topic(i)}")


# Preprocess the new document
# new_document = "The quick brown fox jumps over the lazy dog"
# new_text = [word for word in new_document.lower().split()
#             if word not in stopwords]

# # Convert the preprocessed document into a bag-of-words representation
# new_bow = dictionary.doc2bow(new_text)

# # Infer the topic distribution for the new document
# new_topics = lda.get_document_topics(new_bow)

# # Print the most likely topic for the new document
# most_likely_topic_id, _ = max(new_topics, key=lambda x: x[1])
# most_likely_topic_name = lda.print_topic(most_likely_topic_id)
# print(
#     f"Most likely topic for document '{new_document}' is {most_likely_topic_name}")
