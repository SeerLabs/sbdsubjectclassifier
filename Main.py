import string
from collections import Counter, OrderedDict
import pandas as pd
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk as nk
from nltk.stem import WordNetLemmatizer
from DnnModel import DnnModel



class Main:

    def __init__(self,abstracts_path, WE_path,max_len,tf_idf_sorting=None):
        self.abstracts_path = abstracts_path
        self.tf_idf_sorting = tf_idf_sorting
        self.WE_path = WE_path
        self.max_len = max_len

    def number(self,word):
        try:
            float(word)
            return True
        except:
            return False

    def tf_idf_ordering(self, abstract):
        lemmatizer = WordNetLemmatizer()
        trivial_words = stopwords.words('english') + list(string.printable)
        words = set([lemmatizer.lemmatize(word.lower()) for word in nk.word_tokenize(abstract) if
                        word.lower() not in trivial_words and not self.number(word)])
        tf_idf_list = dict()
        if self.tf_idf_sorting:
            final_dict = OrderedDict(sorted(tf_idf_list.items(), key=lambda x: x[1], reverse=True))
        else:
            position_list = dict()
            pos = 0
            for word in words:
                position_list[word] = pos
                pos = pos + 1
            first_dict = OrderedDict(sorted(tf_idf_list.items(), key=lambda x: x[1], reverse=True))
            # print(count)
            unordered_abstract = list(first_dict[:80])
            final_dict = dict()
            for word in unordered_abstract:
                final_dict[word] = position_list[word]
            final_dict = OrderedDict(sorted(final_dict.items(), key=lambda x: x[1], reverse=False))
        return list(final_dict)

    def get_we_model(self, WE_path):
        file = open(WE_path)
        WE_model = dict()
        for line in file:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]], dtype=np.float32)
            WE_model[word] = embedding
        return WE_model


    def main(self):
        abstracts_file = pd.read_csv(self.abstracts_path,index_col=['abstract','labels'])
        abstracts = self.tf_idf_ordering(abstracts_file['abstract'])
        labels = np.array(abstracts_file['label'],dtype=np.int16)
        classes = len(set(labels))
        WE_model = self.get_we_model(self.WE_path)
        model = dnn_model()






    if __name__ == '__main__':
        main()
