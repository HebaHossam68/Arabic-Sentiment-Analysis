import re
import pandas as pd
import numpy as np
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D,Conv2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
from Preprocessing_Functions import*

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



# Loading and preprocessing the training data
data = pd.read_csv('Dataset/train.csv')
X, Y = preprocessing(data)
training_portion = 0.8
train_size = int(len(X) * training_portion)

X_train = X[0:train_size]
Y_train = Y[0:train_size]

validation_articles = X[train_size:]
validation_labels = Y[train_size:]

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
validation_labels = encoder.transform(validation_labels)

print(Y_train)
print(validation_labels)

model = Sequential()
model.add(Embedding(input_dim=44999, output_dim=128, input_length=100))
model.add(Conv1D(32, 3, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
model.add(LSTM(16, dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))
optimizer=Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])



history = model.fit(X_train, Y_train, epochs=10, batch_size=128, validation_data=(validation_articles, validation_labels))


# Evaluate model performance
plot_training_validation(history)
val_acc = history.history['val_accuracy'][-1]
print(f"Final Validation Accuracy: {val_acc}")

# Load and preprocess the test data
test_data = pd.read_csv('Dataset/test _no_label.csv')


test_X, test_ids = preprocess_test(test_data)

# Make predictions on the test data
class_mapping = {idx: label for idx, label in enumerate(encoder.classes_)}

# Make predictions on the test data
predictions = model.predict(test_X)
predicted_labels = [class_mapping[pred.argmax()] for pred in predictions]

# Map predicted labels to sentiment values: -1, 0, 1
#sentiment_mapping = {0: -1, 1: 0, 2: 1}

# Apply sentiment mapping to predicted labels
# predicted_sentiments = [sentiment_mapping.get(label, -1) for label in predicted_labels]
# # Get the sentiment value directly from the predicted labels
# predicted_sentiments = [label + 1 for label in predicted_labels]

predicted_sentiments = []
for pred in predictions:
    if pred[0] > 0.6:
        predicted_sentiments.append(1)
    elif pred[0] < 0.4:
        predicted_sentiments.append(-1)
    else:
        predicted_sentiments.append(0)

print('predicted_sentiments')
print(predicted_sentiments)

# Create submission DataFrame
# Create submission DataFrame
submission_df = pd.DataFrame({'ID': test_ids, 'rating': predicted_sentiments})

# Replace '2' with '-1' in the 'rating' column
submission_df['rating'] = submission_df['rating'].replace(2, -1)

# Ensure exactly 1000 lines in the submission file
submission_df = submission_df.head(1000)

# Save DataFrame to a CSV file
submission_df.to_csv('submission_LSTM.csv', index=False)


