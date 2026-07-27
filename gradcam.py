import tensorflow as tf
import numpy as np
import cv2


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    # Find MobileNetV2 base model
    base_model = model.layers[0]

    # Get the convolution layer
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )


    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        predicted_class = tf.argmax(predictions[0])

        loss = predictions[:, predicted_class]


    grads = tape.gradient(
        loss,
        conv_outputs
    )


    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )


    conv_outputs = conv_outputs[0]


    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)


    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= tf.reduce_max(heatmap)


    return heatmap.numpy()