import csv
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('arabic'))
import matplotlib.pyplot as plt

def plot_training_validation(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

vocab_size = 5000
embedding_dim = 64
max_length = 200
oov_tok = '<OOV>' #  Out of Vocabulary
training_portion = 0.8

articles = []
labels = []

with open("Dataset/train.csv", 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[1])
        
        article = row[0]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)

print(articles)
train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index


train_sequences = tokenizer.texts_to_sequences(train_articles)

train_padded = pad_sequences(train_sequences, maxlen=max_length,)

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length)



label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

from keras.layers import Dropout, SimpleRNN

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(SimpleRNN(units=embedding_dim))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.summary()



model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

num_epochs = 20
history = model.fit(train_padded, training_label_seq, epochs=num_epochs,
                    validation_data=(validation_padded, validation_label_seq))
plot_training_validation(history)



test_articles = []
test_ids = []

with open("Dataset/test _no_label.csv", 'r', encoding='utf-8') as testfile:
    test_reader = csv.reader(testfile, delimiter=',')
    next(test_reader)  # Skip header
    for row in test_reader:
        test_ids.append(row[0])
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        test_articles.append(article)

test_sequences = tokenizer.texts_to_sequences(test_articles)
test_padded = pad_sequences(test_sequences, maxlen=max_length)

# Make predictions on the test data
predictions = model.predict(test_padded)

# Map predicted labels to sentiment values: 1, 0, -1

predicted_labels = np.argmax(predictions, axis=1)
# sentiment_mapping = {0: -1, 1: 0, 2: 1}
# predicted_sentiments = [sentiment_mapping[label] for label in predicted_labels]
# predicted_sentiments = [label - 1 for label in predicted_labels]
predicted_sentiments = []
for pred in predictions:
    if pred[0] > 0.6:
        predicted_sentiments.append(1)
    elif pred[0] < 0.4:
        predicted_sentiments.append(-1)
    else:
        predicted_sentiments.append(0)

# Create a DataFrame for submission
submission_df = pd.DataFrame({'ID': test_ids, 'rating': predicted_sentiments})

# Ensure exactly 1000 lines in the submission file
submission_df = submission_df.head(1000)

# Save DataFrame to a CSV file with required formatting
submission_df.to_csv('submission.csv', index=False)

