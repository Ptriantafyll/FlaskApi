
# ? here will be the code for bert transformer
import unicodedata
import torch
from transformers import *


# # ? preprocess data
# def strip_accents_and_lowercase(s):
#     return ''.join(c for c in unicodedata.normalize('NFD', s)
#                    if unicodedata.category(c) != 'Mn').lower()


# # Load model and tokenizer
# tokenizer_greek = AutoTokenizer.from_pretrained(
#     'nlpaueb/bert-base-greek-uncased-v1')
# lm_model_greek = AutoModelWithLMHead.from_pretrained(
#     'nlpaueb/bert-base-greek-uncased-v1')

# # ================ EXAMPLE 1 ================
# text_1 = 'O ποιητής έγραψε ένα [MASK] .'
# # EN: 'The poet wrote a [MASK].'
# input_ids = tokenizer_greek.encode(text_1)
# print(tokenizer_greek.convert_ids_to_tokens(input_ids))
# # ['[CLS]', 'o', 'ποιητης', 'εγραψε', 'ενα', '[MASK]', '.', '[SEP]']
# outputs = lm_model_greek(torch.tensor([input_ids]))[0]
# print(tokenizer_greek.convert_ids_to_tokens(outputs[0, 5].max(0)[1].item()))
# # the most plausible prediction for [MASK] is "song"

# # ================ EXAMPLE 2 ================
# text_2 = 'Είναι ένας [MASK] άνθρωπος.'
# # EN: 'He is a [MASK] person.'
# input_ids = tokenizer_greek.encode(text_2)
# print(tokenizer_greek.convert_ids_to_tokens(input_ids))
# # ['[CLS]', 'ειναι', 'ενας', '[MASK]', 'ανθρωπος', '.', '[SEP]']
# outputs = lm_model_greek(torch.tensor([input_ids]))[0]
# print(tokenizer_greek.convert_ids_to_tokens(outputs[0, 3].max(0)[1].item()))
# # the most plausible prediction for [MASK] is "good"

# # ================ EXAMPLE 3 ================
# text_3 = 'Είναι ένας [MASK] άνθρωπος και κάνει συχνά [MASK].'
# # EN: 'He is a [MASK] person he does frequently [MASK].'
# input_ids = tokenizer_greek.encode(text_3)
# print(input_ids)
# print(tokenizer_greek.convert_ids_to_tokens(input_ids))
# # ['[CLS]', 'ειναι', 'ενας', '[MASK]', 'ανθρωπος', 'και', 'κανει', 'συχνα', '[MASK]', '.', '[SEP]']
# outputs = lm_model_greek(torch.tensor([input_ids]))[0]
# print(tokenizer_greek.convert_ids_to_tokens(outputs[0, 8].max(0)[1].item()))
# # the most plausible prediction for the second [MASK] is "trips"

# text_4 = 'The poet wrote a [MASK].'
# input_ids = tokenizer_greek.encode(text_4)
# print(input_ids)
# print(tokenizer_greek.convert_ids_to_tokens(input_ids))
# # ['[CLS]', 'o', 'ποιητης', 'εγραψε', 'ενα', '[MASK]', '.', '[SEP]']
# outputs = lm_model_greek(torch.tensor([input_ids]))[0]
# print(outputs)
# print(tokenizer_greek.convert_ids_to_tokens(outputs[0, 8].max(0)[1].item()))
# # the most plausible prediction for [MASK] is "song"

from transformers import BertTokenizer, BertModel

# Load the tokenizer and model for the multilingual BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
model = BertModel.from_pretrained("bert-base-multilingual-uncased")

# Example English text
english_text = "This is an example sentence in English."

# Example Greek text
greek_text = "Αυτή είναι μια πρόταση στα Ελληνικά."

# Tokenize the input texts
english_inputs = tokenizer(
    english_text, return_tensors="pt", padding=True, truncation=True)
greek_inputs = tokenizer(greek_text, return_tensors="pt",
                         padding=True, truncation=True)

# Forward pass through the model for English text
english_outputs = model(**english_inputs)
english_embeddings = english_outputs.last_hidden_state[:, 0, :]

# Forward pass through the model for Greek text
greek_outputs = model(**greek_inputs)
greek_embeddings = greek_outputs.last_hidden_state[:, 0, :]

print("English embeddings:")
print(english_embeddings)

print("Greek embeddings:")
print(greek_embeddings)
