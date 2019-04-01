import nltk
import numpy as np
import random
import string

# Term Frequency-Inverse Document Frequency (TF-IDF)
# This bag of words heuristic weighs word scores based on the total number of docs
# over how many docs the word appears in, effectively discarding biases for popular
# words like 'the'.  
from sklearn.feature_extraction.text import TfidVectorizer

# Cosine similarity
# Used to find the similarity between user input and words in the corpora
from sklearn.metrics.pairwise import cosine_similarity

# ELIZA just uses keyword matching for greetings:
USER_GREETINGS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "heyo", "what up", "yo")
AGENT_RESPONSES = ("hello, human", "hi", "oh.. hi there", "hello", "hi... I'm a little shy", "hello... I'm not very good at conversations")

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



