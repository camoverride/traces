import tensorflow as tf

def create_alpha_blending_model(height, width, channels):
    input_1 = tf.keras.Input(shape=(None, height, width, channels), name='new_frames')
    input_2 = tf.keras.Input(shape=(None, height, width, channels), name='current_frames')
    
    alpha = 0.5  # or use another method to determine alpha dynamically
    blended_output = alpha * input_1 + (1 - alpha) * input_2
    
    model = tf.keras.Model(inputs=[input_1, input_2], outputs=blended_output)
    return model

# Example usage
height, width, channels = 1920, 1080, 3
model = create_alpha_blending_model(height, width, channels)
model.summary()


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('alpha_blending_batch_model.tflite', 'wb') as f:
    f.write(tflite_model)
