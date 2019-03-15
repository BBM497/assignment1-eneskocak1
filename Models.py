import string
import collections
import numpy as np

class Essay(object):

    # Class Attribute
    species = 'Essay'

    # Initializer / Instance Attributes
    def __init__(self, essay_path, tokenizer=True):
        self.essay_path = essay_path
        self.file = open(essay_path, "r")
        self.author = self.file.readline().rstrip(' \n')
        self.essay = self.file.readline().rstrip(' \n').lower()
        self.file.close()
        self.words = ""
        if tokenizer:
            self.tokenizer()
        self.splitter()

    # instance method
    def information(self):
        return "This essay written by {} ".format(self.author)

    # tokenizer for performing
    def tokenizer(self):
        self.essay = self.essay.replace(".", " END END . START START ")


    # splitting function
    def splitter(self):
        self.essay = self.essay.translate(str.maketrans("", "", string.punctuation))
        self.words = self.essay.split(" ")


class Model(object):

    # Class Attribute
    species = "Language Model"

    # Used Model Name
    active_model = "None"

    # Initial False, Controlling for is Created Already?
    isModelCreated = False

    # Initializer / Instance Attributes
    def __init__(self, author):
        self.essays = []
        self.uni_model_words = []
        self.bi_model_words = []
        self.tri_model_words = []
        self.author = author
        self.uni_bag_of_words = dict()
        self.bi_bag_of_words = dict()
        self.tri_bag_of_words = dict()
        self.all_words = []

    # Adding Essay to Language Model
    def add_essay(self, new_essay):
        self.essays.append(new_essay)
        self.all_words += new_essay.words

    # Creating unigram, bigram and trigram models and creating bagOfWords
    def create_models(self):
        if not self.isModelCreated:
            self.isModelCreated = True
            self.unigram_model()
            self.bigram_model()
            self.trigram_model()
            self.get_bag_of_words()
        else:
            print("You already Created Models")

    # Creating Unigram Model
    def unigram_model(self):
        for essay in self.essays:
            self.uni_model_words += essay.words

    # Creating Bigram Model
    def bigram_model(self):
        for essay in self.essays:
            for i, j in zip(essay.words[0::1], essay.words[1::1]):
                self.bi_model_words.append(i+" "+j)

    # Creating trigram Model
    def trigram_model(self):
        for essay in self.essays:
            for i, j, k in zip(essay.words[0::1], essay.words[1::1], essay.words[2::1]):
                self.tri_model_words.append(i + " " + j + " " + k)

    # Creating BagOfWord for each language model
    def get_bag_of_words(self):
        self.uni_bag_of_words = dict(collections.Counter(self.uni_model_words))
        self.bi_bag_of_words = dict(collections.Counter(self.bi_model_words))
        self.tri_bag_of_words = dict(collections.Counter(self.tri_model_words))

    # Getting Model Probabilities

    def get_probabilities(self,testword,mode):
        if mode == "bigram":
            keywords = testword.rsplit(" ", 1)
            print(keywords)
            pro = (self.bi_bag_of_words.get(testword, 0)+1)/(self.uni_bag_of_words.get(keywords[0], 0)+self.bi_bag_of_words.keys().__len__())
            return -np.log10(pro)
        if mode == "trigram":
            keywords = testword.rsplit(" ", 1)
            pro = (self.tri_bag_of_words.get(testword, 0)+1)/(self.bi_bag_of_words.get(keywords[0], 0)+self.tri_bag_of_words.keys().__len__())
            return -np.log10(pro)
        if mode == "unigram":
            keywords = testword.rsplit(" ", 1)
            pro = (self.uni_bag_of_words.get(testword, 0)+1)/(len(self.uni_model_words)+self.uni_bag_of_words.keys().__len__())
            return -np.log10(pro)

    def perplexity(self):
        print("enes")