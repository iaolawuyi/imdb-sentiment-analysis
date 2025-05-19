import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from utils import vectorize_sequences

# load the train data
(train_data, train_labels), _ = imdb.load_data(num_words=1000)

# Preprocess train and test data
x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype(np.float32)

# Split into train and validation data
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# create callback for model saving
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            os.path.dirname(__file__), 
            "../saved_models/model_v1.keras"),
        save_best_only=True
    )
]

# Model Definition
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=4,
    batch_size=512,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

# Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "../graphs/graph_v1.jpg"))
plt.show()