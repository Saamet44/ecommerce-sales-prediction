
from flask import Flask, jsonify
import numpy as np
import tensorflow as tf
import pickle

# Model ve scaler yükleme
model = tf.keras.models.load_model('../model/lstm_model.h5')
with open('../model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict_sales():
    # Örnek bir test girdisi
    sequence_length = 10
    test_input = np.random.rand(sequence_length, 1).reshape(1, sequence_length, 1)  # Dummy input
    prediction = model.predict(test_input)
    scaled_prediction = scaler.inverse_transform(prediction)
    return jsonify({'predicted_sales_quantity': float(scaled_prediction[0][0])})

if __name__ == '__main__':
    app.run(debug=True)
