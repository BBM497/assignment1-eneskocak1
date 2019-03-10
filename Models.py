import string


class Essay:

    # Class Attribute
    species = 'essay'

    # Initializer / Instance Attributes
    def __init__(self, essay_path, tokenizer=False):
        self.essay_path = essay_path
        self.file = open(essay_path, "r")
        self.path = essay_path
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


class Model:

    species = "Language Model"

    def __init__(self, author):
        self.essays = []
        self.all_uniq_words = set()
        self.author = author
        self.word_count = 0

    def add_essay(self, new_essay):
        self.essays.append(new_essay)
        # print("essay added success")

    def uniq_word(self, modelname = "unigram"):
        self.all_uniq_words = set()
        if modelname == "unigram":
            self.all_uniq_words |= self.unigram_model()
        if modelname == "bigram":
            self.all_uniq_words |= self.bigram_model()
        if modelname == "trigram":
            self.all_uniq_words |= self.trigram_model()

    def unigram_model(self):
        print("unigram activated")
        uniq_words = set()
        self.word_count = 0
        for essay in self.essays:
            uniq_words |= set(essay.words)
            self.word_count += len(essay.words)
        return uniq_words

    def bigram_model(self):
        print("bigram activated")
        uniq_words = set()
        self.word_count = 0
        for essay in self.essays:
            for i, j in zip(essay.words[0::1], essay.words[1::1]):
                uniq_words.add(i+" "+j)
                self.word_count += 1
        return uniq_words

    def trigram_model(self):
        print("trigram activated")
        uniq_words = set()
        self.word_count = 0

        for essay in self.essays:
            for i, j, k in zip(essay.words[0::1], essay.words[1::1], essay.words[2::1]):
                uniq_words.add(i + " " + j + " " + k)
                self.word_count += 1
        return uniq_words
