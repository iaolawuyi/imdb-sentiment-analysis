from tensorflow import keras
from preprocess import load_data

_, (x_test, y_test) = load_data()

model = keras.models.load_model("../saved_models/model_v1.keras")

results = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {results[1]*100:.2f}%")