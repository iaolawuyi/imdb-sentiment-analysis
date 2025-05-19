import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from utils import vectorize_sequences

model_path = os.path.join(os.path.dirname(__file__), "../saved_models/model_v1.keras")

# load the test data
_, (test_data, test_labels) = imdb.load_data(num_words=1000)
x_test = vectorize_sequences(test_data)
y_test = np.asarray(test_labels).astype(np.float32)

# Load the model
model = keras.models.load_model(os.path.abspath(model_path))

# evaluate the model

loss, acc = model.evaluate(x_test, y_test)
print(f"Loss: {loss:.2f}")
print(f"Accuracy: {acc:.2f}")