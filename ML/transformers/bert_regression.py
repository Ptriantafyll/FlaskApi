import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
import json
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from transformers import TFDistilBertForSequenceClassification
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# This is needed to import a function from different directory
import sys
sys.path.append(r"C:\Users\ptria\source\repos\FlaskApi\ML")
from functions import normalize, preprocess
import matrix_factorization

# Returns user-url matrix as pandas DataFrame
df = matrix_factorization.perform_martix_factorization()
# pick a user from pandas df
user = df.index[1]

# ? File that contains all the urls in the mongodb cluster
url_file = open(
    r"C:\Users\ptria\source\repos\FlaskApi\Web-Scraping\json\urls.json", encoding="utf-8")
urls = json.load(url_file)

# ? Load pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")

# ? distilbert base model with pretrained weights
pretrained_bert = TFDistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")


dataset_length = len(df.columns)
# ? use max length 128 instead of the max length of all sequences
max_length = 128
documents_input_ids = np.zeros(
    (dataset_length, max_length))  # ids of the bert tokenizer
# masks are 0 if they are the result of padding
documents_masks = np.zeros((dataset_length, max_length))
# todo: change iterations and get data from pandas df
counter = 0
user_ratings = []
for url in urls:
    # ? test with just 10 rows to see if my nn is capable
    if counter == dataset_length:
        break
    text = url["text"]
    text = preprocess.clean_text(text)
    text = text.lower()
    text = preprocess.filter_sentences_english_and_greek(text)

    # encode all documents
    tokenized_text = tokenizer.encode_plus(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    documents_input_ids[counter, :] = tokenized_text.input_ids
    documents_masks[counter, :] = tokenized_text.attention_mask
    counter = counter + 1

    user_ratings.append(df.loc[user, url['url']] - 1)  # ratings 0-4 from 1-5


ratings = np.array(user_ratings)
ratings = ratings.astype(np.int32)

documents_input_ids = documents_input_ids.astype(np.int32)
documents_masks = documents_masks.astype(np.int32)


# Train-val
X_train, X_final_test, y_train, y_final_test, train_mask, final_test_mask = train_test_split(
    documents_input_ids, ratings, documents_masks, test_size=0.1, random_state=42)

# Train-test
X_train, X_test, y_train, y_test, train_mask, test_mask = train_test_split(
    X_train, y_train, train_mask, test_size=0.2, random_state=42)


# Convert y_train to a Python list
y_train_list = y_train

class_weights = compute_class_weight(
    'balanced', classes=np.unique(y_train_list), y=y_train_list)
class_weights_dict = dict(enumerate(class_weights))

# defining 2 input layers for input_ids and attn_masks
input_ids = tf.keras.layers.Input(
    shape=(max_length,), name='input_ids', dtype='int32')
attn_masks = tf.keras.layers.Input(
    shape=(max_length,), name='attention_mask', dtype='int32')

bert_embds = pretrained_bert.distilbert(
    input_ids=input_ids, attention_mask=attn_masks)

output = bert_embds[0]
output = output[:, 0, :]


num_of_neurons = 1024
dropout_percent = 0.4
l2_regularizer_weight = 5e-4
output = tf.keras.layers.Dense(num_of_neurons, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.L2(l2_regularizer_weight))(output)
output = tf.keras.layers.Dropout(dropout_percent)(output)
output = tf.keras.layers.Dense(num_of_neurons, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.L2(l2_regularizer_weight))(output)
output = tf.keras.layers.Dropout(dropout_percent)(output)
output = tf.keras.layers.Dense(num_of_neurons, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.L2(l2_regularizer_weight))(output)
output = tf.keras.layers.Dropout(dropout_percent)(output)
output = tf.keras.layers.Dense(num_of_neurons, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.L2(l2_regularizer_weight))(output)
output = tf.keras.layers.Dropout(dropout_percent)(output)
output = tf.keras.layers.Dense(1, activation='linear')(output)

rating_model = tf.keras.models.Model(
    inputs=[input_ids, attn_masks], outputs=output)

for layer in rating_model.layers[:3]:
    layer.trainable = False

# Print model
rating_model.summary()

lr = 3e-4
optimizer = tf.keras.optimizers.Nadam(
    learning_rate=lr
)

# Train model
rating_model.compile(
    optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

num_of_epochs = 100
# To add class weight: class_weight=class_weights_dict
history = rating_model.fit(
    [X_train, train_mask],
    y_train,
    batch_size=32,
    epochs=num_of_epochs,
    validation_data=([X_test, test_mask], y_test),
    class_weight=class_weights_dict
)

# Save model
rating_model.save('rating_model')


predictions = rating_model.predict([X_test, test_mask])
predicted_classes = [max(0, min(round(x), 4)) for x in predictions.flatten()]
# Compute the confusion matrix
cm = confusion_matrix(y_test, predicted_classes)

# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(r"C:\Users\ptria\source\repos\FlaskApi\images\val_cm.png")
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.savefig(r"C:\Users\ptria\source\repos\FlaskApi\images\mse.png")
plt.legend()
plt.show()

# Plot training and validation MAE
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.savefig(r"C:\Users\ptria\source\repos\FlaskApi\images\mae.png")
plt.show()

predictions = rating_model.predict([X_train, train_mask])
predicted_classes = [max(0, min(round(x), 4)) for x in predictions.flatten()]
# Compute the confusion matrix
cm = confusion_matrix(y_train, predicted_classes)

# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(r"C:\Users\ptria\source\repos\FlaskApi\images\train_cm.png")
plt.show()

# Evaluate the model on the test data
loss, mae = rating_model.evaluate(
    [X_final_test, final_test_mask], y_final_test)

predictions = rating_model.predict([X_final_test, final_test_mask])
predicted_classes = [max(0, min(round(x), 4)) for x in predictions.flatten()]
# Compute the confusion matrix
cm = confusion_matrix(y_final_test, predicted_classes)

# Display the confusion matrix using seaborn for better visualization
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(r"C:\Users\ptria\source\repos\FlaskApi\images\test_cm.png")
plt.show()

# print("Parameters for this run\n\n")
# print("Optimizer: ", optimizer)
# print("Learning rate: ", lr)
# print("Neurons: ", num_of_neurons)
# print("Dropout: ", dropout_percent)
# print("L2 Regularizer weight:", l2_regularizer_weight)
# print("Epochs: ", num_of_epochs)
# print("Max length: ", max_length)
# print("transformer: ", pretrained_bert)
