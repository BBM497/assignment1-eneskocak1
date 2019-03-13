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
Hamilton.get_probabilities()
#Madison.uniq_word(modelname="unigram")



