import nltk
import numpy as np
import random
import string

# ELIZA just uses keyword matching for greetings:
USER_GREETINGS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "heyo", "what up", "yo")
AGENT_RESPONSES = ("hello, human", "hi", "oh.. hi there", "hello", "Hi... I'm a little shy", "Hello... I'm not very good at talking with humans")

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

def greeting(userText):
    for word in userText.split():
        if word.lower() in USER_GREETINGS:
            return random.choice(AGENT_RESPONSES)

