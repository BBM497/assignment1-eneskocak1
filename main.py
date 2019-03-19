import settings
import Models as model


def classification_test(test_indexes, modelname, generate=False, test=False):

    Hamilton = model.Model("Hamilton", modelname)
    Madison = model.Model("Madison", modelname)

    Hamilton.add_essay(settings.hamilton_train_essays_indexes)
    Madison.add_essay(settings.madison_train_essays_indexes)
    if generate:
        for i in range(2):
            Hamilton.generator()
            Madison.generator()
    if test:
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
                print(str(path)+".txt "+Unk.active_model.upper(), "WIN:[HAMILTON] Hamilton:", "{0:.5f}".format(hamilton_prob), "Madison:", "{0:.5f}".format(madison_prob))
            else:
                print(str(path)+".txt "+Unk.active_model.upper(), "WIN:[MADISON] Hamilton:", "{0:.5f}".format(hamilton_prob), "Madison:", "{0:.5f}".format(madison_prob))




classification_test(settings.hamilton_test_essays_indexes, "unigram",generate=True,test=True)
print("\n")
classification_test(settings.hamilton_test_essays_indexes, "bigram",generate=True,test=True)
print("\n")
classification_test(settings.hamilton_test_essays_indexes, "trigram",generate=True,test=True)



