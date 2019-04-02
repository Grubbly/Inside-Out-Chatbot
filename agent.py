# ELIZA just uses keyword matching for greetings:
import nltk
import numpy as np
import random
import string

# Term Frequency-Inverse Document Frequency (TF-IDF)
# This bag of words heuristic weighs word scores based on the total number of docs
# over how many docs the word appears in, effectively discarding biases for popular
# words like 'the'.  
from sklearn.feature_extraction.text import TfidfVectorizer

# Cosine similarity
# Used to find the similarity between user input and words in the corpora
from sklearn.metrics.pairwise import cosine_similarity

USER_GREETINGS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "heyo", "what up", "yo")
AGENT_RESPONSES = ("hello, human", "hi", "oh.. hi there", "hello", "hi... I'm a little shy", "hello... I'm not very good at conversations")

class Agent:
    def __init__(
        self, 
        name=None, 
        corpus=None, 
        greetingMessage=None, 
        defaultMessage=None, 
        goodbyeMessage=None
    ):
        if name is None:
            self.name = "Chatbot"
        else:
            self.name = name
        
        if corpus is None:
            corpusFile = open('./corpora/default.txt', 'r', errors='ignore')
            self.corpus = corpusFile.read().lower()
        else:
            corpusFile = open('./corpora/'+corpus, 'r', errors='ignore')
            self.corpus = corpusFile.read().lower()

        if greetingMessage is None:
            self.greetingMessage = self.name + ": My name is " + self.name + " The Chatbot!\nIf you want to leave, type 'bye'"
        else:
            self.greetingMessage = self.name + ": " + greetingMessage

        if defaultMessage is None:
            self.defaultMessage = "Sorry, I don't understand."
        else:
            self.defaultMessage = defaultMessage

        if goodbyeMessage is None:
            self.goodbyeMessage = self.name + ": See ya later!"
        else:
            self.goodbyeMessage = self.name + ": " + goodbyeMessage

        nltk.download('punkt')
        nltk.download('wordnet')

        self.corpusSentences = nltk.sent_tokenize(self.corpus)
        self.corpusWords = nltk.word_tokenize(self.corpus)

        # Tokenization and normalization handlers
        self.lemmer = nltk.stem.WordNetLemmatizer()
        self.removePunctuation = dict((ord(punct), None) for punct in string.punctuation)

        self.introduction()


    def introduction(self):
        print("\n\n")
        print("**********************************")
        print("***** Inside Out Chatbot CLI *****")
        print("**********************************")
        print("\n")
        print(self.greetingMessage)

    def lemanizeTokens(self, tokens):
        return [self.lemmer.lemmatize(token) for token in tokens]

    def normalize(self, text):
        return self.lemanizeTokens(nltk.word_tokenize(text.lower().translate(self.removePunctuation)))

    def greeting(self, userText):
        for word in userText.split():
            if word.lower() in USER_GREETINGS:
                return random.choice(AGENT_RESPONSES)

    def response(self, userText):
        agentResponse=''
        self.corpusSentences.append(userText)
        
        # Stop words are words that do not contribute to the understanding of text
        # Here, we are using a predefined list of such words.
        TfidfVector = TfidfVectorizer(tokenizer=self.normalize, stop_words='english')
        termFrequencies = TfidfVector.fit_transform(self.corpusSentences)
        similarities = cosine_similarity(termFrequencies[-1], termFrequencies)
        corpusSentencesIndex = similarities.argsort()[0][-2]
        flat = similarities.flatten()
        flat.sort()
        resultantTermFrequency = flat[-2]

        if(resultantTermFrequency == 0):
            agentResponse = agentResponse + self.defaultMessage
            return agentResponse
        else:
            agentResponse = agentResponse + self.corpusSentences[corpusSentencesIndex]
            return agentResponse

    def sense(self):
        userInput = input()
        userInput = userInput.lower()

    def chatCLI(self):
        chatting = True
        while chatting:
            userInput = input("Chat: ")
            userInput = userInput.lower()

            if(userInput != "bye"):
                if(self.greeting(userInput) != None):
                    print(self.name + ": " + self.greeting(userInput))
                else:
                    print(self.name + ": ", end="")
                    print(self.response(userInput) + "\n")
                    self.corpusSentences.remove(userInput)
            else:
                chatting = False
                print(self.goodbyeMessage)