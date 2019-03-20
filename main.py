import settings
import Models as model
import datetime

def classification_test(test_indexes, test_author, modelname, generate=False, test=False):

    Hamilton = model.Model("Hamilton", modelname)
    Madison = model.Model("Madison", modelname)

    Hamilton.add_essay(settings.hamilton_train_essays_indexes)
    Madison.add_essay(settings.madison_train_essays_indexes)
    if generate:
        for i in range(2):
            Hamilton.generator()
            Madison.generator()
    if test:
        f = open("results/classification/"+test_author + "/" + modelname + "_test_results.txt", "w+")
        for path in test_indexes:
            Unk = model.Model("UNK", modelname)
            Unk.add_essay([path])
            hamilton_prob =0
            madison_prob = 0
            model_words = Unk.get_model_words()
            for i in model_words:
                hamilton_prob += Hamilton.get_probabilities(i)
                madison_prob += Madison.get_probabilities(i)

            hamilton_prob = Unk.perplexity(hamilton_prob, len(model_words))
            madison_prob = Unk.perplexity(madison_prob, len(model_words))
            if hamilton_prob < madison_prob:
                result = str(path)+".txt "+Unk.active_model.upper() +" WIN:[HAMILTON] Hamilton Perplexity: "+ "{0:.5f}".format(hamilton_prob) + " Madison Perplexity:"+" {0:.5f}".format(madison_prob)

            else:
                result = str(path)+".txt "+Unk.active_model.upper()+" WIN:[MADISON] Hamilton Perplexity: "+ "{0:.5f}".format(hamilton_prob)+ " Madison Perplexity:"+ " {0:.5f}".format(madison_prob)

            print(result)
            f.write(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ") + result + "\n")

        f.close()



# HAMİLTON CLASSIFICATION TEST PART
classification_test(settings.hamilton_test_essays_indexes, "Hamilton", "unigram", generate=False, test=True)
print("\n")
classification_test(settings.hamilton_test_essays_indexes, "Hamilton", "bigram", generate=False, test=True)
print("\n")
classification_test(settings.hamilton_test_essays_indexes, "Hamilton", "trigram", generate=False, test=True)
print("\n")

# MADISON CLASSIFICATION TEST PART
classification_test(settings.madison_test_essays_indexes, "Madison", "unigram", generate=False, test=True)
print("\n")
classification_test(settings.madison_test_essays_indexes, "Madison", "bigram", generate=False, test=True)
print("\n")
classification_test(settings.madison_test_essays_indexes, "Madison", "trigram", generate=False, test=True)
print("\n")

# UNKNOWN CLASSIFICATION TEST PART
classification_test(settings.test_essays_indexes, "Unknown", "unigram", generate=False, test=True)
print("\n")
classification_test(settings.test_essays_indexes, "Unknown", "bigram", generate=False, test=True)
print("\n")
classification_test(settings.test_essays_indexes, "Unknown", "trigram", generate=False, test=True)

# GENERATE ESSAY TEST PART
classification_test(settings.test_essays_indexes, "Unknown", "unigram", generate=True, test=False)
print("\n")
classification_test(settings.test_essays_indexes, "Unknown", "bigram", generate=True, test=False)
print("\n")
classification_test(settings.test_essays_indexes, "Unknown", "trigram", generate=True, test=False)
print("\n")