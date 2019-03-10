import settings
import Models as model

Hamilton = model.Model("Hamilton")
Madison = model.Model("Madison")

for i in settings.hamilton_train_essays_indexes:
    new = model.Essay(settings.federalist_papers_directory+str(i)+".txt", tokenizer=False)
    Hamilton.add_essay(new)

for i in settings.madison_train_essays_indexes:
    new = model.Essay(settings.federalist_papers_directory+str(i)+".txt")
    Madison.add_essay(new)


Hamilton.uniq_word(modelname="trigram")
Madison.uniq_word(modelname="unigram")
print(Hamilton.all_uniq_words.__len__(), Hamilton.word_count)
print(Madison.all_uniq_words.__len__(), Madison.word_count)

