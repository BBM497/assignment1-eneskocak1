import string
import collections


class Essay(object):

    # Class Attribute
    species = 'essay'

    # Initializer / Instance Attributes
    def __init__(self, essay_path, tokenizer=False):
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
        self.essay = self.essay.replace(". ", " [DOTINHERE] ")
        # self.essay = self.essay.replace(",", " [SEPERATORISHERE]")

    # spitting function
    def splitter(self):
        self.essay = self.essay.translate(str.maketrans("", "", string.punctuation))
        self.words = self.essay.split(" ")


class Model(object):

    species = "Language Model"
    active_model = "None"
    isModelCreated = False

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

    def add_essay(self, new_essay):
        self.essays.append(new_essay)
        self.all_words += new_essay.words

    def create_models(self):
        if not self.isModelCreated:
            self.isModelCreated = True
            self.unigram_model()
            self.bigram_model()
            self.trigram_model()
            self.get_bag_of_words()
        else:
            print("You already Created Models")

    def unigram_model(self):
        for essay in self.essays:
            self.uni_model_words += essay.words

    def bigram_model(self):
        for essay in self.essays:
            for i, j in zip(essay.words[0::1], essay.words[1::1]):
                self.bi_model_words.append(i+" "+j)

    def trigram_model(self):
        for essay in self.essays:
            for i, j, k in zip(essay.words[0::1], essay.words[1::1], essay.words[2::1]):
                self.tri_model_words.append(i + " " + j + " " + k)

    def get_bag_of_words(self):
        self.uni_bag_of_words = dict(collections.Counter(self.uni_model_words))
        self.bi_bag_of_words = dict(collections.Counter(self.bi_model_words))
        self.tri_bag_of_words = dict(collections.Counter(self.tri_model_words))

    def get_probabilities(self):
        for i,j in self.tri_bag_of_words.items():
            keywords = i.rsplit(" ", 1)
            print("P("+keywords[1]+"|"+keywords[0]+") = ", j/self.bi_bag_of_words[keywords[0]])
