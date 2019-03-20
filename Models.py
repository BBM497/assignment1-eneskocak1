import string
import collections
import numpy as np
import settings
import random as rd
import datetime
class Essay(object):

    # Class Attribute
    species = 'Essay'
    tokenlist = [".", "?", "!", ":", ",", "-", ";", "(", ")", "[", "]", "`", "'", "<", ">", "/", "@", "%"]
    # Initializer / Instance Attributes
    def __init__(self, essay_path, modelname, tokenizer=True):
        self.essay_path = essay_path
        self.file = open(essay_path, "r")
        self.author = self.file.readline().rstrip(' \n')
        self.essay = self.file.readline().rstrip(' \n').lower()
        self.file.close()
        self.words = []
        self.modelname = modelname
        if tokenizer:
            self.tokenizer()
        self.splitter()

    # instance method
    def information(self):
        return "This essay written by {} ".format(self.author)

    # tokenizer for performing
    def tokenizer(self):
        if self.modelname == "unigram":
            self.essay = self.essay.replace(".", " . ")
            self.essay = self.essay.replace("?", " ? ")
            self.essay = self.essay.replace("!", " ! ")
        if self.modelname == "bigram":
            self.essay = self.essay.replace(".", settings.SENTENCE_END+"."+settings.SENTENCE_START)
            self.essay = self.essay.replace("?", settings.SENTENCE_END+"?"+settings.SENTENCE_START)
            self.essay = self.essay.replace("!", settings.SENTENCE_END+"!"+settings.SENTENCE_START)
        if self.modelname == "trigram":
            self.essay = self.essay.replace(".", settings.SENTENCE_END + settings.SENTENCE_END + "." + settings.SENTENCE_START+settings.SENTENCE_START)
            self.essay = self.essay.replace("?", settings.SENTENCE_END + settings.SENTENCE_END + "?" + settings.SENTENCE_START+settings.SENTENCE_START)
            self.essay = self.essay.replace("!", settings.SENTENCE_END + settings.SENTENCE_END + "!" + settings.SENTENCE_START+settings.SENTENCE_START)
        self.essay = self.essay.replace(":", " : ")
        self.essay = self.essay.replace(",", " , ")
        self.essay = self.essay.replace("-", " - ")
        self.essay = self.essay.replace(";", " ; ")
        self.essay = self.essay.replace("(", " ( ")
        self.essay = self.essay.replace(")", " ) ")
        self.essay = self.essay.replace("[", " [ ")
        self.essay = self.essay.replace("]", " ] ")
        self.essay = self.essay.replace("`", " ` ")
        self.essay = self.essay.replace("'", " ' ")
        self.essay = self.essay.replace("@", " @ ")
        self.essay = self.essay.replace("%", " % ")
    # splitting function
    def splitter(self):
        if self.modelname == "bigram":
            self.essay = settings.SENTENCE_START + self.essay
        if self.modelname == "trigram":
            self.essay = settings.SENTENCE_START + settings.SENTENCE_START + self.essay

        self.words = self.essay.split(" ")
        while True:
            try:
                self.words.remove("")
            except:
                break
        if self.modelname == "bigram":
            if self.words[-1] == "<s>":
                self.words.pop(-1)
        if self.modelname == "trigram":
            for i in range(1,3):
                if self.words[-1] == "<s>":
                    self.words.pop(-1)

class Model(object):

    # Class Attribute
    species = "Language Model"

    # Used Model Name
    active_model = "None"

    # Initial False, Controlling for is Created Already?
    isModelCreated = False

    # Initializer / Instance Attributes
    def __init__(self, author, model_name):
        self.essays = []
        self.uni_model_words = []
        self.bi_model_words = []
        self.tri_model_words = []
        self.author = author
        self.uni_bag_of_words = dict()
        self.bi_bag_of_words = dict()
        self.tri_bag_of_words = dict()
        self.all_words = []
        self.active_model = model_name

    # Adding Essay to Language Model
    def add_essay(self, indexes):
        for i in indexes:
            new = Essay(settings.federalist_papers_directory+str(i)+".txt", self.active_model)
            self.essays.append(new)
            self.all_words += new.words
        self.create_models()
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

    def get_probabilities(self,testword):
        if self.active_model == "bigram":
            keywords = testword.rsplit(" ", 1)
            pro = np.divide(self.bi_bag_of_words.get(testword, 0)+1, (self.uni_bag_of_words.get(keywords[0], 0)+self.bi_bag_of_words.keys().__len__()))
            return -np.log2(pro)
        if self.active_model == "trigram":
            keywords = testword.rsplit(" ", 1)
            pro = np.divide(self.tri_bag_of_words.get(testword, 0)+1, (self.bi_bag_of_words.get(keywords[0], 0)+self.tri_bag_of_words.keys().__len__()))
            return -np.log2(pro)
        if self.active_model == "unigram":

            keywords = testword.rsplit(" ", 1)
            pro = np.divide(self.uni_bag_of_words.get(testword, 0)+1, (len(self.uni_model_words)+self.uni_bag_of_words.keys().__len__()))
            return -np.log2(pro)

    def get_model_words(self):
        if self.active_model == "unigram":
            return self.uni_model_words
        if self.active_model == "bigram":
            return self.bi_model_words
        if self.active_model == "trigram":
            return self.tri_model_words


    def perplexity(self, number, count):
        return np.power(2, (number/count))

    def generator(self):
        print("*******************************************************************************************************")
        print(
            "This Essay generated for: " + self.author.upper() + " and use with " + self.active_model.upper() + " model -->")
        generate = True
        index = 0
        newessay = "Essay not created please contact with admin :D"
        if self.active_model == "unigram":
            newessay = []
            while generate:
                if index != 0:
                    if index == 29 or newessay[index-1] == "." or newessay[index-1] == "?" or newessay[index-1] == "!":
                        break
                olasi = ({i: j for i, j in self.uni_bag_of_words.items()})
                returned = self.cumulative_returner(olasi)
                newessay.append(returned)
                index += 1

        if self.active_model == "bigram":

            newessay = [settings.SENTENCE_START.strip(" ")]
            while generate:
                if index == 30 or newessay[index] == "." or newessay[index] == "?" or newessay[index] == "!":
                    break
                olasi = ({i: j for i, j in self.bi_bag_of_words.items() if newessay[index] == i.rsplit(" ",1)[0]})
                returned = self.cumulative_returner(olasi)
                newessay.append(returned)
                index += 1
        if self.active_model == "trigram":
            newessay = [settings.SENTENCE_START.strip(" "),settings.SENTENCE_START.strip(" ")]
            index = 1
            while generate:
                if index == 29 or newessay[index] == "." or newessay[index] == "?" or newessay[index] == "!":
                    break
                olasi = ({i: j for i, j in self.tri_bag_of_words.items() if newessay[index-1] + " " + newessay[index] == i.rsplit(" ",1)[0]})
                returned = self.cumulative_returner(olasi)
                newessay.append(returned)
                index += 1


        print(' '.join(newessay))
        print("*******************************************************************************************************")
        f = open("results/generate/"+self.author+"/"+self.active_model+"_generated.txt","a")
        f.write(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")+" ".join(newessay)+"\n")
        f.close()

    def cumulative_returner(self,dict):

        total = (np.sum(list(dict.values())))
        start = 0
        cumulative_dict = {}
        for i,j in dict.items():
            cumulative_dict[i]={"start": start, "end":start+(j/total)}
            start += j/total
        random = rd.random()
        for i,j in cumulative_dict.items():
            if random >= j["start"] and random < j["end"]:
                if self.active_model == "unigram":
                    return i
                else:
                    return i.rsplit(" ",1)[1]
