import tensorflow as tf
from tensorflow.keras.layers import Layer

class OverlayLayer(Layer):
    def __init__(self, alpha=0.5, **kwargs):
        super(OverlayLayer, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs):
        input1, input2 = inputs
        output = tf.add(tf.multiply(self.alpha, input1), tf.multiply(1 - self.alpha, input2))
        return output

def create_overlay_model(alpha=0.5):
    input1 = tf.keras.layers.Input(shape=(None, None, 3), name="input_1")
    input2 = tf.keras.layers.Input(shape=(None, None, 3), name="input_2")
    
    overlay_output = OverlayLayer(alpha=alpha)([input1, input2])
    
    model = tf.keras.models.Model(inputs=[input1, input2], outputs=overlay_output)
    return model

# Create the model
alpha = 0.5  # Replace with your ALPHA value
overlay_model = create_overlay_model(alpha)

# Save the model
overlay_model.save("overlay_model.h5")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(overlay_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model
with open("overlay_model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved as overlay_model.tflite")
