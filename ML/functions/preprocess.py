import re
import json
from nltk.corpus import stopwords


# Removes stop words from a given text
def remove_stopwords(text):

    # ? take greek stop words from file
    greek_stop_words_file = open(
        r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\greek_stop_words.json", encoding="utf8")
    greek_stop_words = json.load(greek_stop_words_file)
    # take english stopwords from nltk
    stop_words = set(stopwords.words('english'))

    filtered_words = [
        w for w in text.split() if not w in stop_words and not w in greek_stop_words and len(w) > 1]

    # Join the filtered words into a single string
    result = ' '.join(filtered_words)

    return result


def clean_text(temp):
    temp = re.sub("@\S+", " ", temp)
    temp = re.sub("https*\S+", " ", temp)
    temp = re.sub("#\S+", " ", temp)
    temp = re.sub("\'\w+", '', temp)
    temp = re.sub(r'\w*\d+\w*', '', temp)
    temp = re.sub('\s{2,}', " ", temp)
    temp = re.sub('[^A-Za-z ]+', '', temp)
    return temp.strip()


# Removes all non english and non greek words from a text
def filter_sentences_english_and_greek(text):
    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Define patterns for English and Greek words
    english_pattern = re.compile(r'[a-zA-Z]+')
    greek_pattern = re.compile(r'[α-ωΑ-Ωά-ώΆ-Ώ]+')

    # Filter sentences based on language
    filtered_sentences = []
    for sentence in sentences:
        english_words = re.findall(english_pattern, sentence)
        greek_words = re.findall(greek_pattern, sentence)

        if english_words or greek_words:
            filtered_sentences.append(sentence)

    # Join the filtered sentences into a single string
    result = ' '.join(filtered_sentences)

    return result
