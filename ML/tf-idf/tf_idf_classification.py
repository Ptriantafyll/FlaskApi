import pickle

# Load the vectorizer from the file
with open(r"C:\Users\ptria\source\repos\FlaskApi\ML\tf-idf\tfidf_vectorizer.pkl", 'rb') as f:
    loaded_tfidf_vectorizer = pickle.load(f)


new_text = ["This is a new document to be vectorized."]
new_text_tfidf = loaded_tfidf_vectorizer.transform(new_text)
print(new_text_tfidf)
