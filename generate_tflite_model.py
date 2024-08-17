import tensorflow as tf

def alpha_blending_model(input_shape):
    input_1 = tf.keras.Input(shape=input_shape)
    input_2 = tf.keras.Input(shape=input_shape)
    
    # Alpha blending operation
    alpha = 0.5  # This is the alpha value used in your original code
    output = alpha * input_1 + (1 - alpha) * input_2
    
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
    return model

# Define the input shape (Height, Width, Channels)
input_shape = (1920, 1080, 3)  # (Height, Width, Channels)

# Create the model
model = alpha_blending_model(input_shape)

# Save the model as a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a file
with open("alpha_blending_model.tflite", "wb") as f:
    f.write(tflite_model)
