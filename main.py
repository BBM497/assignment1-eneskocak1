import settings
import Models as model
import copy


Hamilton = model.Model("Hamilton")
Madison = model.Model("Madison")

for i in settings.hamilton_train_essays_indexes:
    new = model.Essay(settings.federalist_papers_directory+str(i)+".txt")
    Hamilton.add_essay(new)

for i in settings.madison_train_essays_indexes:
    new = model.Essay(settings.federalist_papers_directory+str(i)+".txt")
    Madison.add_essay(new)



Hamilton.create_models()
Madison.create_models()

def test(path,modelname):
    test = model.Essay(settings.federalist_papers_directory +str(path)+ ".txt")
    Unk= model.Model("UNK")
    Unk.add_essay(test)
    Unk.create_models()
    total =0
    madi = 0
    basemodel= "None"
    if modelname == "unigram":
        basemodel = Unk.uni_model_words
    if modelname == "bigram":
        basemodel = Unk.bi_model_words
    if modelname == "trigram":
        basemodel = Unk.tri_model_words
    for i in basemodel:
        total += Hamilton.get_probabilities(i, modelname)
        madi += Madison.get_probabilities(i, modelname)

    total = 2**(total/len(basemodel))
    madi = 2 **(madi / len(basemodel))

    if total > madi:
        print("Hamilton probabilitie: ",total,"\nMadison probabilitie: ",madi)
        print("This is HAMILTON")
    else:
        print("Hamilton probabilitie: ", total, "\nMadison probabilitie: ", madi)
        print("This is MADISON")

for i in settings.hamilton_test_essays_indexes:
    test(i,"trigram")