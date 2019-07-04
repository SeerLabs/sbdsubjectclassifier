import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import multi_gpu_model, to_categorical
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub

embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")


class MlpModelWithUSE:

    def __init__(self, abstracts_path, nodes=128, layers=4, loss='categorical_crossentropy',
                 optimizer='Adam', activation='relu', dropout='0.2', batch_size=1000, epochs=50,
                 gpus=None):
        self.abstracts_path = abstracts_path
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

    def MlpModel(self, nodes, layers, classes, loss, optimizer, activation, input_shape, dropout, gpus):
        model_mlp = Sequential()
        model_mlp.add(Dense(nodes, input_dim=input_shape, activation=activation))
        model_mlp.add(Dropout(dropout))
        for i in range(layers - 2):
            model_mlp.add(Dense(nodes, activation=activation))
            model_mlp.add(Dropout(dropout))
        model_mlp.add(Dense(classes))
        model_mlp.add(Activation(tf.nn.softmax))
        if gpus is None:
            model_mlp.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
            return model_mlp
        model_gpu = multi_gpu_model(model_mlp, gpus=self.gpus)
        model_gpu.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model_gpu

    def mini_batch_generator(self, train_data):
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            return session.run(embed(train_data))

    def train(self, batch_size, epochs, X_train, y_train, classes, model):
        for n_epoch in range(epochs):
            batch_length = int(len(X_train) / batch_size)
            for batch in range(batch_length + 1):
                if batch == batch_length:
                    x1 = batch * batch_size
                    train_data = X_train[x1:]
                    label_data = y_train[x1:]
                else:
                    x1 = batch * batch_size
                    x2 = (batch + 1) * batch_size
                    train_data = X_train[x1:x2]
                    label_data = y_train[x1:x2]
                model.train_on_batch(self.mini_batch_generator(train_data),
                                     to_categorical(label_data, classes))
        return model

    def test(self, X_test, batch_size, model):
        y_pred = np.array([])
        batch_length = int(len(X_test) / batch_size)
        for batch in range(batch_length + 1):
            if batch == batch_length:
                x1 = batch * batch_size
                test_data = X_test[x1:]
            else:
                x1 = batch * batch_size
                x2 = (batch + 1) * batch_size
                test_data = X_test[x1:x2]
            y_pred = np.append(y_pred, model.predict(self.mini_batch_generator(test_data)))
        y_pred = y_pred.argmax(axis=-1)
        y_pred = np.reshape(y_pred, (len(X_test), 1))
        return y_pred

    def main(self):
        abstracts_file = pd.read_csv(self.abstracts_path, index_col=['abstract', 'labels'])
        abstracts = abstracts_file['abstract']
        labels = np.array(abstracts_file['label'], dtype=np.int16)
        classes = len(set(labels))
        X_train, X_test, y_train, y_test = train_test_split(abstracts, labels, stratify=labels, test_size=0.1,
                                                            random_state=42)
        model = self.MlpModel(self.nodes, self.layers, classes, self.loss, self.optimizer, self.activation, 512,
                              self.dropout, self.gpus)
        model_dnn = self.train(self.batch_size, self.epochs, X_train, y_train, classes, model)
        y_pred = self.test(X_test, self.batch_size, model_dnn)
        print(f1_score(y_test, y_pred, average='micro'))
        print(recall_score(y_test, y_pred, average='micro'))
        print(precision_score(y_test, y_pred, average='micro'))
        print(accuracy_score(y_test, y_pred))

    if __name__ == '__main__':
        main()
