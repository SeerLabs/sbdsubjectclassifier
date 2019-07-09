from __future__ import print_function
import string
from collections import OrderedDict
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk as nk
from nltk.corpus import wordnet
from idf_score_calculator import IDFScoreCalculator


class tfidf_ordering:

    def __init__(self,data_path,tfidf_sorting=True,max_len=80):
        self.tfidf_sorting = tfidf_sorting
        self.data_path = data_path
        self.max_len = max_len
        self.idf_weights = IDFScoreCalculator(data_path)

    def number(self,word):
        try:
            float(word)
            return True
        except:
            return False

    def get_wordnet_pos(self,word):
        tag = nk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)


    def tf_idf_ordering(self, abstract):
        lemmatizer = WordNetLemmatizer()
        trivial_words = stopwords.words('english') + list(string.printable)
        words = set([lemmatizer.lemmatize(word.lower()) for word in nk.word_tokenize(abstract) if
                     word.lower() not in trivial_words and not self.number(word)])

        tf_idf_list = dict()
        for word in words:
            try:
                tf_idf_list[word] = self.idf_weights[word]
            except:
                print('in except')
                tf_idf_list[word] = 0

        if self.tfidf_sorting:
            final_dict = OrderedDict(sorted(tf_idf_list.items(), key=lambda x: x[1], reverse=True))
        else:
            position_list = dict()
            pos = 0
            for word in words:
                position_list[word] = pos
                pos = pos + 1
            first_dict = OrderedDict(sorted(tf_idf_list.items(), key=lambda x: x[1], reverse=True))
            # print(count)
            unordered_abstract = list(first_dict[:self.max_len])
            final_dict = dict()
            for word in unordered_abstract:
                final_dict[word] = position_list[word]
            final_dict = OrderedDict(sorted(final_dict.items(), key=lambda x: x[1], reverse=False))
        return list(final_dict)

    def main(self):
        data = pd.read_csv(self.data_path)
        data.columns = ['abstract', 'labels']
        final_list = list(map(lambda x: list(self.tf_idf_ordering(x)), data['abstract']))
        ordered_list = []
        for abstract in final_list:
            ordered_list.append(" ".join(abstract))
        data['abstract'] = ordered_list
        data.to_csv('final_tfidf_ordered_data.csv')






