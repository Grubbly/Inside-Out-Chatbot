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