from __future__ import print_function
import string
from collections import Counter
import pandas as pd
from nltk.corpus import stopwords
import math
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib
import nltk as nk
from nltk.corpus import wordnet


class IDFScoreCalculator:

    def __init__(self,data_path):
        self.data_path = data_path
        self.main()


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


    def main(self):
        all_abstracts = pd.read_csv(self.data_path)
        file_count = len(all_abstracts)
        trivial_words = stopwords.words('english') + list(string.printable)
        lemmatizer = WordNetLemmatizer()
        final_list = list(map(lambda x: list(set([lemmatizer.lemmatize(word.lower(),
                     self.get_wordnet_pos(word.lower())) for word in nk.word_tokenize(x) if
                     word.lower() not in trivial_words and not self.number(word)])), all_abstracts))

        flatten = [item for sublist in final_list for item in sublist]
        idf_z = dict(Counter(flatten))
        print(len(idf_z))
        print(file_count)
        for key in idf_z:
            idf_z[key] = math.log10(file_count / idf_z[key])

        print('model completed')

        try:
            joblib.dump(idf_z, 'idf_weights.pkl')
        except:
            pass

        pass

    if __name__ == '__main__':
        main()