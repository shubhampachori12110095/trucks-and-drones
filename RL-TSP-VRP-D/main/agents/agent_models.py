import tensorflow as tf
from tensorflow.keras import layers


class BaseLayers(tf.keras.Model):

    def __init__(self,
            model_layer = None,
            n_hidden = 0,
            units_hidden = 32,
            activation_hidden = 'relu',
    ):
        super().__init__()

        if model_layer is None:
            self.model_layer = [layers.Dense(units_hidden, activation=activation_hidden) for i in range(n_hidden)]
        else:
            self.model_layer = model_layer

        self.flatten = layers.Flatten()

    def add_output_layer(self, n, activation_out):

        if activation_out is None:
            self.model_layer.append(layers.Dense(n))
        else:
            self.model_layer.append(layers.Dense(n, activation=activation_out))

    def call(self, inputs: tf.Tensor):

        if len(inputs) > 1:
            inputs = self.flatten(inputs)

        x = inputs
        for l in self.model_layer: x = l(x)
        return x
