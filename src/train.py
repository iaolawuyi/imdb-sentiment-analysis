import os
from tensorflow.keras.callbacks import EarlyStopping
from preprocess import load_data
from model import build_model
from plot import plot_history

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

def main():
    (x_train, y_train), _ = load_data()

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    model = build_model()
    
    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=10,
        batch_size=512,
        validation_data=(x_val, y_val),
        callbacks = [early_stopping]
    )

    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/model_v2.keras")

    os.makedirs("graphs", exist_ok=True)
    plot_history(history, "graphs/graph_v2.jpg")

if __name__ == "__main__":
    main()
