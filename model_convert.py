import tensorflow as tf
import onnx
import keras2onnx

# Load the Keras model
keras_model = tf.keras.models.load_model("./model/model.weights.h5")

# Convert to ONNX
onnx_model = keras2onnx.convert_keras(keras_model, keras_model.name)

# Save ONNX model
onnx.save_model(onnx_model, "./model/efficientnetv2.onnx")

print("Converted Keras model to ONNX format")

