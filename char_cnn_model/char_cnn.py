from __future__ import print_function
from __future__ import division
import string
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, classification_report, accuracy_score
import cnn_model
import numpy as np
import pandas as pd
from cnn_model import cnn_model

class char_cnn:

    def __init__(self, data_path, batch_size=1000,epochs=50, gpus= None):
        self.abstracts_path = data_path
        self.batch_size = batch_size
        self.nb_filter = 256
        self.dense_outputs = 1024
        self.filter_kernels = [7, 7, 3, 3, 3, 3]
        self.gpus = gpus
        self.epochs = epochs
        self.main()

    def main(self):
        abstracts_file = pd.read_csv(self.abstracts_path, index_col=['abstract', 'labels'])
        abstracts = abstracts_file['abstract']
        labels = np.array(abstracts_file['label'], dtype=np.int16)
        classes = len(set(labels))
        alphabet = set(list(string.ascii_lowercase) + list(string.digits) +
                       list(string.punctuation) + ['\n'])
        vocab_size = len(alphabet)
        vocab = dict()
        for ix, t in enumerate(alphabet):
            vocab[t] = ix

        X_train, X_test, y_train, y_test = train_test_split(abstracts, labels, stratify=labels, test_size=0.1,
                                                            random_state=42)

        model = cnn_model(self.filter_kernels, self.dense_outputs,vocab_size,
                                           self.nb_filter, classes,self.batch_size,vocab,gpus=self.gpus)

        model_dnn = model.create_model()
        model_dnn = model.train(self.batch_size, self.epochs, X_train, y_train, classes,
                                model_dnn)
        y_pred = model.test(X_test, self.batch_size, model_dnn)
        print(f1_score(y_test, y_pred, average='micro'))
        print(recall_score(y_test, y_pred, average='micro'))
        print(precision_score(y_test, y_pred, average='micro'))
        print(accuracy_score(y_test, y_pred))

    if __name__ == '__main__':
        main()

