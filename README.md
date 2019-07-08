# sbdsubjectclassifier:Scholarly Big Data Subject Category Classifier

In this project, we attempt to classify research papers into their subject areas. By splitting abstracts into words and converting each word into a n-dimensional word embedding,
we project the text data into a vector space with time-steps. In order to select the words, we use TF-IDF weights to find the most important words in the abstract and sort them based on the TF-IDF values. 
To classify the data, we use 2 flavors of Recurrent Neural networks (RNN's) namely : LSTM and GRU. We tried various Word Embedding (WE) models such as
GloVe, SciBert and FastText. We also use Universal Sentence Encoder (USE) with Multi-layers perceptron (MLP) and char-level CNN to compare the performance of the above model.

### Requirements:
1. Keras 
2. tensorflow (tensorflow_gpu recommended)
3. nltk
4. pandas
5. sklearn
6. tensorflow_hub(for USE)

### Models
&nbsp;   &nbsp;   &nbsp; [RNNs with WE](https://github.com/SeerLabs/sbdsubjectclassifier/tree/master/keras_model):
This model takes text data path (.csv file with data and label) and  WE file path as inputs and classifies the data using RNN's. 

#### step1: clean the data. 
In order to clean and order the data, use [tf_idf sorting module](https://github.com/SeerLabs/sbdsubjectclassifier/blob/master/tfidf_ordering/tfidf_ordering.py). 
This module splits the sentences into words and removes stopwords, punctuations and numbers from the words. These words are lemmatized and then sorted based on TF-IDF values.
This module takes 3 arguments:
1. data_path : path to the text data
2. max_len  : maximum length of the words to be retained in each text sequence (in our case, abstract).
3. tf_idf ordering (optional) : boolean value. set to `True' to sort the values based on TF-IDF valuses. 'False' retains the order instead of sorting.

The cleaned data will be saved in a csv file named 'final_tfidf_ordered_data.csv'.

