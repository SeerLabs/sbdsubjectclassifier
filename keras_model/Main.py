import os
import string
from collections import OrderedDict
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk as nk
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from keras_model.RnnModel import RnnModels

from keras_model.RnnModelsGpu import RnnModelsGpu

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class Model:

    def __init__(self, abstracts_path, WE_path, max_len=80, nodes=128, layers=2, loss='categorical_crossentropy',
                 optimizer='Adam', activation='tanh', dropout='0.2', batch_size=1000, epochs=50, tf_idf_sorting=True,
                 gpus=None):
        self.abstracts_path = abstracts_path
        self.tf_idf_sorting = tf_idf_sorting
        self.WE_path = WE_path
        self.max_len = max_len
        self.nodes = nodes
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.activation = activation
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.gpus = gpus
        self.main()

    def number(self, word):
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
        abstracts_file = pd.read_csv(self.abstracts_path, index_col=['abstract', 'labels'])
        abstracts = self.tf_idf_ordering(abstracts_file['abstract'])
        labels = np.array(abstracts_file['label'], dtype=np.int16)
        classes = len(set(labels))
        WE_model = self.get_we_model(self.WE_path)
        X_train, X_test, y_train, y_test = train_test_split(abstracts, labels, stratify=labels, test_size=0.1,
                                                            random_state=42)
        input_shape = (self.max_len, len(WE_model[next(iter(WE_model))]))

        if self.gpus is None:
            model = RnnModels(self.nodes, self.layers, classes, self.loss, self.optimizer, self.activation, input_shape,
                              self.dropout)
        else:
            model = RnnModelsGpu(self.nodes, self.layers, classes, self.loss, self.optimizer, self.activation,
                                 input_shape,self.dropout, self.gpus)

        model_dnn = model.get_bigru_model()
        model_dnn = model.train(self.batch_size, self.epochs, X_train, y_train, self.max_len, WE_model, classes,
                                model_dnn)
        y_pred = model.test(X_test, self.batch_size, model_dnn, self.max_len, WE_model)
        print(f1_score(y_test, y_pred, average='micro'))
        print(recall_score(y_test, y_pred, average='micro'))
        print(precision_score(y_test, y_pred, average='micro'))
        print(accuracy_score(y_test, y_pred))

    if __name__ == '__main__':
        main()
