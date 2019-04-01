import nltk
import numpy as np
import random
import string

corpusFile = open('../corpora/sadPoems.txt', 'r', errors='ignore')
corpus = corpusFile.read()
corpus = corpus.lower()

nltk.download('punkt')
nltk.download('wordnet')

corpusSentences = nltk.sent_tokenize(corpus)
corpusWords = nltk.word_tokenize(corpus)

# Tokenization and normalization globals
lemmer = nltk.stem.WordNetLemmatizer()
removePunctuation = dict((ord(punct), None) for punct in string.punctuation)

def lemanizeTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def normalize(text):
    return lemanizeTokens(nltk.word_tokenize(text.lower().translate(removePunctuation)))