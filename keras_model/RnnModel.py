from keras import Sequential
from keras.layers import Bidirectional, GRU, Dropout, Dense, LSTM
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import nltk as nk


class RnnModels:

    def __init__(self,nodes, layers, classes,loss,optimizer,activation,input_shape,dropout):
        self.nodes = nodes
        self.layers = layers
        self.classes = classes
        self.loss = loss
        self.optimizer = optimizer
        self.activation = activation
        self.input_shape = input_shape
        self.dropout = dropout


    def get_bigru_model(self):
        model_dnn = Sequential()
        model_dnn.add(Bidirectional(GRU(self.nodes, return_sequences=True,activation=self.activation),
                                    input_shape=self.input_shape))
        model_dnn.add(Dropout(self.dropout))

        for i in range(int(self.layers-2)):
            model_dnn.add(Bidirectional(GRU(self.nodes, return_sequences=True,activation=self.activation)))
            model_dnn.add(Dropout(self.dropout))
        model_dnn.add(Bidirectional(GRU(self.nodes, return_sequences=False,activation=self.activation)))
        model_dnn.add(Dropout(self.dropout))
        model_dnn.add(Dense(self.classes, activation='sigmoid'))
        model_dnn.compile(loss= self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model_dnn

    def get_bilstm_model(self):
        model_dnn = Sequential()
        model_dnn.add(Bidirectional(LSTM(self.nodes, return_sequences=True,activation=self.activation),
                                    input_shape=self.input_shape))
        model_dnn.add(Dropout(self.dropout))
        for i in range(int(self.layers - 2)):
            model_dnn.add(Bidirectional(LSTM(self.nodes, return_sequences=True,activation=self.activation)))
            model_dnn.add(Dropout(self.dropout))
        model_dnn.add(Bidirectional(LSTM(self.nodes, return_sequences=False,activation=self.activation)))
        model_dnn.add(Dropout(self.dropout))
        model_dnn.add(Dense(self.classes, activation='sigmoid'))
        model_dnn.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model_dnn

    def get_gru_model(self):
        model_dnn = Sequential()
        model_dnn.add(GRU(self.nodes, return_sequences=True, input_shape=self.input_shape,
                           activation=self.activation))
        model_dnn.add(Dropout(self.dropout))

        for i in range(int(self.layers - 2)):
            model_dnn.add(GRU(self.nodes, return_sequences=True, activation=self.activation))
            model_dnn.add(Dropout(self.dropout))
        model_dnn.add(GRU(self.nodes, return_sequences=False, activation=self.activation))
        model_dnn.add(Dropout(self.dropout))
        model_dnn.add(Dense(self.classes, activation='sigmoid'))
        model_dnn.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model_dnn

    def get_lstm_model(self):
        model_dnn = Sequential()
        model_dnn.add(LSTM(self.nodes, return_sequences=True, input_shape=self.input_shape,
                                    activation=self.activation))
        model_dnn.add(Dropout(self.dropout))

        for i in range(int(self.layers - 2)):
            model_dnn.add(LSTM(self.nodes, return_sequences=True,activation=self.activation))
            model_dnn.add(Dropout(self.dropout))
        model_dnn.add(LSTM(self.nodes, return_sequences=False,activation=self.activation))
        model_dnn.add(Dropout(self.dropout))
        model_dnn.add(Dense(self.classes, activation='sigmoid'))
        model_dnn.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        return model_dnn

    def mini_batch_generator(self, X_mini, max_len,WE_model):
        data = []
        count_new = 0
        count_in = 0
        # X_mini = list(map(lambda x: list(tf_idf_ordering(x)), X_mini))
        for abstract in X_mini:
            abstract = nk.word_tokenize(abstract)
            feature = []
            count = 0
            for word in abstract:
                if count >= max_len:
                    break
                word_new = word.lower()
                if word_new in WE_model:
                    count = count + 1
                    feature.append(WE_model[word_new])
                else:
                    count_new = count_new + 1
            data.append(list(feature))
            count_in = count_in + count
            # if count_new > 0:
        # print(count_in, count_new)
        data1 = pad_sequences(data, padding='post', maxlen=max_len)
        return np.array(data1, dtype=np.float16)

    def train(self,batch_size,epochs,X_train, y_train,max_len,WE_model,classes,model):
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
                model.train_on_batch(self.mini_batch_generator(train_data,max_len, WE_model),
                                         to_categorical(label_data, classes))
        return model

    def test(self, X_test,batch_size,model,max_len,WE_model):
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
            y_pred = np.append(y_pred, model.predict_classes(self.mini_batch_generator(test_data,max_len, WE_model)))
        return y_pred








