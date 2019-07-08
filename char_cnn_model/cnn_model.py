from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.initializers import RandomNormal
import tensorflow as tf
from keras.utils import multi_gpu_model, to_categorical
import numpy as np


class cnn_model:

    def __init__(self, filter_kernels, dense_outputs,vocab_size, nb_filter, classes, batch_size,vocab, max_len= 1014,gpus=None):
        self.filter_kernels = filter_kernels
        self.dense_outputs = dense_outputs
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.nb_filter = nb_filter
        self.classes = classes
        self.gpus = gpus
        self.batch_size = batch_size
        self.vocab = vocab

    def one_hot(self, x):
        return tf.one_hot(x, self.vocab_size, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32)

    def one_hot_outshape(self, in_shape):
        return in_shape[0], in_shape[1], self.vocab_size

    def create_model(self):
        initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)

        # Define what the input shape looks like
        inputs = Input(shape=(self.max_len,), dtype='int64')

        embedded = Lambda(self.one_hot, output_shape=self.one_hot_outshape)(inputs)

        # All the convolutional layers...
        conv = Convolution1D(filters=self.nb_filter, kernel_size=self.filter_kernels[0], kernel_initializer=initializer,
                             padding='valid', activation='relu',
                             input_shape=(self.max_len, self.vocab_size))(embedded)
        conv = MaxPooling1D(pool_size=3)(conv)

        conv1 = Convolution1D(filters=self.nb_filter, kernel_size=self.filter_kernels[1],
                              kernel_initializer=initializer,
                              padding='valid', activation='relu')(conv)
        conv1 = MaxPooling1D(pool_size=3)(conv1)

        conv2 = Convolution1D(filters=self.nb_filter, kernel_size=self.filter_kernels[2],
                              kernel_initializer=initializer,
                              padding='valid', activation='relu')(conv1)

        conv3 = Convolution1D(filters=self.nb_filter, kernel_size=self.filter_kernels[3],
                              kernel_initializer=initializer,
                              padding='valid', activation='relu')(conv2)

        conv4 = Convolution1D(filters=self.nb_filter, kernel_size=self.filter_kernels[4],
                              kernel_initializer=initializer,
                              padding='valid', activation='relu')(conv3)

        conv5 = Convolution1D(filters=self.nb_filter, kernel_size=self.filter_kernels[5],
                              kernel_initializer=initializer,
                              padding='valid', activation='relu')(conv4)
        conv5 = MaxPooling1D(pool_size=3)(conv5)
        conv5 = Flatten()(conv5)

        # Two dense layers with dropout of .5
        z = Dropout(0.5)(Dense(self.dense_outputs, activation='relu')(conv5))
        z = Dropout(0.5)(Dense(self.dense_outputs, activation='relu')(z))

        # Output dense layer with softmax activation
        pred = Dense(self.classes, activation='softmax', name='output')(z)

        model = Model(inputs=inputs, outputs=pred)

        adam = Adam(lr=0.001)
        if self.gpus != None:
            model = multi_gpu_model(model, gpus=self.gpus)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        return model

    def encode_data(self, x, maxlen, vocab):
        # Iterate over the loaded data and create a matrix of size (len(x), maxlen)
        # Each character is encoded into a one-hot array later at the lambda layer.
        # Chars not in the vocab are encoded as -1, into an all zero vector.

        input_data = np.zeros((len(x), maxlen), dtype=np.int)
        for dix, sent in enumerate(x):
            counter = 0
            for c in sent:
                if c == ' ':
                    continue
                if counter >= maxlen:
                    pass
                else:
                    ix = vocab.get(c, -1)  # get index from vocab dictionary, if not in vocab, return -1
                    if c == ' ':
                        print(ix, 'space')
                    input_data[dix, counter] = ix
                    counter += 1
                return input_data

    def train(self,batch_size,epochs,X_train, y_train,classes,model):
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
                model.train_on_batch(self.encode_data(train_data, self.max_len, self.vocab), to_categorical(label_data, classes))
        return model

    def test(self, X_test,batch_size,model):
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
            y_pred = np.append(y_pred, model.predict_classes(self.encode_data(test_data, self.max_len, self.vocab)))
            return y_pred
