import numpy as np
from tensorflow.keras.datasets import imdb

def decode_to_text(data, index):
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    decoded_review = " ".join(
        [reverse_word_index.get(i - 3, "?") for i in data[index]])
    
    return decoded_review

def vectorize_sequences(sequences, dimensions=10000):
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.

    return results