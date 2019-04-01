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

NAME = "Saddie"

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

def response(userText):
    agentResponse=''
    corpusSentences.append(userText)
    
    # Stop words are words that do not contribute to the understanding of text
    # Here, we are using a predefined list of such words.
    TfidfVector = TfidVectorizer(tokenizer=normalize, stop_words='english')
    termFrequencies = TfidfVector.fit_transform(corpusSentences)
    similarities = cosine_similarity(termFrequencies[-1], termFrequencies)
    corpusSentencesIndex = similarities.argsort()[0][-2]
    flat = similarities.flatten()
    flat.sort()
    resultantTermFrequency = flat[-2]

    if(resultantTermFrequency == 0):
        agentResponse = agentResponse + "Sorry.. I don't know what to say :("
        return agentResponse
    else:
        agentResponse = agentResponse + corpusSentences[corpusSentencesIndex]
        return agentResponse
        

chatting = True
print(NAME + ": My name is " + NAME + " The Sadbot. I'm an expert in sadness :(")
print("If you want to leave, type 'bye'")

while(chatting):
    userInput = input()
    userInput = userInput.lower()

    if(userInput != "bye"):
        if(greeting(userInput) != None):
            print(NAME + ": " + greeting(userInput))
        else:
            print(NAME + ": ", end="")
            print(response(userInput))
            corpusSentences.remove(userInput)
    else:
        chatting = False
        print(NAME + ": Goodbye :(")
        