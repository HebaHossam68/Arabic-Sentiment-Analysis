import re
import pandas as pd
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def remove_punctuation(text):
    punc_pattern = r'(?<=\w)([^\w\s] |_) +(?=\w)'
    text = re.sub(punc_pattern, '', text)
    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocessing(data):
    X = data['review_description']
    Y = data['rating']
    stop_words = set()
    stop_words.update(set(stopwords.words('arabic')))
    stop_words.update(set(stopwords.words('english')))
    lemmatizer = WordNetLemmatizer()
    X = X.apply(remove_punctuation)
    X = X.apply(lambda x: " ".join(x for x in word_tokenize(x)))
    X = X.apply(lambda x: " ".join(x for x in word_tokenize(x) if x not in stop_words))
    X = X.apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

    all_words = ' '.join(X).split()
    unique_words = set(all_words)

    # Calculate the length of unique words
    length_unique_words = len(unique_words)
    print(f"Length of unique words: {length_unique_words}")
    tokenizer = Tokenizer(num_words=44999)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)

    return pad_sequences(sequences, maxlen=100), Y

def preprocess_test(data):
    X_test = data['review_description']
    stop_words = set(stopwords.words('arabic'))
    stop_words.update(set(stopwords.words('english')))
    lemmatizer = WordNetLemmatizer()
    X_test = X_test.apply(remove_punctuation)
    X_test = X_test.apply(lambda x: " ".join(x for x in word_tokenize(x)))
    X_test = X_test.apply(lambda x: " ".join(x for x in word_tokenize(x) if x not in stop_words))
    X_test = X_test.apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))

    all_words_test = ' '.join(X_test).split()
    unique_words_test = set(all_words_test)

    # Calculate the length of unique words in test data
    length_unique_words_test = len(unique_words_test)
    print(f"Length of unique words in test data: {length_unique_words_test}")
    tokenizer = Tokenizer(num_words=44999)
    tokenizer.fit_on_texts(X_test)
    sequences = tokenizer.texts_to_sequences(X_test)

    return pad_sequences(sequences, maxlen=100), data['ID']