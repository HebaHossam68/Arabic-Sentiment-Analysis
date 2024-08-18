import re
import pandas as pd
import numpy as np
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D


def remove_punctuation(text):
    punc_pattern = r'(?<=\w)([^\w\s] |_) +(?=\w) '
    text = re.sub(punc_pattern, '', text)
    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text


def preprocessing(data):

    stop_words = set()
    stop_words.update(set(stopwords.words('arabic')))
    stop_words.update(set(stopwords.words('english')))
    lemmatizer = WordNetLemmatizer()
    data['review_description'] = data['review_description'].apply(remove_punctuation)
    data['review_description'] = data['review_description'].apply(lambda x: " ".join(x for x in word_tokenize(x)))
    data['review_description'] = data['review_description'].apply(
        lambda x: " ".join(x for x in word_tokenize(x) if x not in stop_words))
    data['review_description'] = data['review_description'].apply(
        lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

    return data


data = pd.read_csv('Dataset/train.csv')

preprocessed_data = preprocessing(data)

label_mapping = {1: 0,
                 -1: 1,
                 0: 2} 
preprocessed_data['rating'] = preprocessed_data['rating'].map(label_mapping)

unique_classes = preprocessed_data['rating'].unique()
print(unique_classes)

num_classes = len(unique_classes)
print(preprocessed_data.head(5))

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(data['review_description'])
tfidf_values = tfidf_vect.transform(data['review_description'])

X_train,X_val,Y_train,Y_val=train_test_split(tfidf_values, preprocessed_data['rating'], test_size=0.2, random_state=42)
max_len = 100  
X_train = pad_sequences(X_train.toarray(), maxlen=max_len)
X_val = pad_sequences(X_val.toarray(), maxlen=max_len)


model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, epochs=2, batch_size=64, validation_data=(X_val, Y_val))



test_data = pd.read_csv('Dataset/test _no_label.csv')  
processed_test_data = preprocessing(test_data)
test_tfidf_values = tfidf_vect.transform(processed_test_data['review_description'])
test_tfidf_values = pad_sequences(test_tfidf_values.toarray(), maxlen=max_len)
print(test_tfidf_values)
predictions = model.predict(test_tfidf_values)
print(predictions)



reverse_label_mapping = {v: k for k, v in label_mapping.items()}
predicted_labels = [reverse_label_mapping[np.argmax(pred)] for pred in predictions]

# Create a DataFrame for submission
submission_df = pd.DataFrame({'ID': test_data['ID'], 'rating': predicted_labels})

# Save DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)





