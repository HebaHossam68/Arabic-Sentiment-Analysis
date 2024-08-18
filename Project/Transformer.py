import re
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential 
from keras.layers import Embedding, Dense, Flatten, Dropout, Attention, LayerNormalization
from keras.optimizers import Adam
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from Preprocessing_Functions import *
 
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
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
att = Attention(use_scale=True, dropout=0.1)
ffn = Sequential([Dense(64, activation="relu"), Dense(128),])
layernorm1 = LayerNormalization(epsilon=1e-6)
layernorm2 = LayerNormalization(epsilon=1e-6)
dropout1 = tf.keras.layers.Dropout(0.1)
dropout2 = tf.keras.layers.Dropout(0.1)
inputs = model.layers[-1].output
attn_output = att([inputs, inputs])
attn_output = dropout1(attn_output)
out1 = layernorm1(inputs + attn_output)
ffn_output = ffn(out1)
ffn_output = dropout2(ffn_output)
transformer_output = layernorm2(out1 + ffn_output)
max_len = 500
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
model.fit(X_train, Y_train, epochs=2, batch_size= 128, validation_data=(validation_articles, validation_labels))
 

test_data = pd.read_csv('Dataset/test _no_label.csv')
X_test, test_ids = preprocess_test(test_data)
 
predictions = model.predict(X_test)
predicted_labels = [encoder.classes_[pred.argmax()] for pred in predictions]
 
sentiment_mapping = {0: -1, 1: 0, 2: 1}
predicted_sentiments = [sentiment_mapping.get(label, -1) for label in predicted_labels]
 
submission_df = pd.DataFrame({'ID': test_ids, 'rating': predicted_sentiments})
submission_df = submission_df.head(1000)
submission_df.to_csv('Transformer_submission1.csv', index=False)