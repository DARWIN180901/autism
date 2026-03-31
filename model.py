import tensorflow as tf
# Convert Image Model
model = tf.keras.models.load_model('model_1/optimized_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
with open('model/optimized_model.tflite', 'wb') as f:
    f.write(converter.convert())