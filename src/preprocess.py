import os
import numpy as np
from tensorflow.keras.datasets import imdb
from utils import vectorize_sequences

def load_data(num_words=10000):
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
    
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    
    y_train = np.asarray(train_labels).astype(np.float32)
    y_test = np.asarray(test_labels).astype(np.float32)
    
    return (x_train, y_train), (x_test, y_test)