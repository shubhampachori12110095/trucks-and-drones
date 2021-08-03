import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

class ParallelGradientTape:

    def __init__(
            self,
            optimizer: optimizer_v2.OptimizerV2 = None,
            grad_alpha: float = 1.0,
            dynamic_grad_alpha: bool = False,
            tanh_grad_aplha: bool = False,
            dga_rate: float = 0.015
    ):

        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam()
        else:
            self.optimizer = optimizer

        self.grad_alpha = tf.constant(grad_alpha, dtype=tf.float32)

        self.dynamic_grad_alpha = dynamic_grad_alpha
        if self.dynamic_grad_alpha:
            self.dga_rate = tf.constant(dga_rate, dtype=tf.float32)
            self.one_minus_dga_rate = tf.constant(1 - dga_rate, dtype=tf.float32)

        self.tanh_grad_aplha = tanh_grad_aplha

        self.name = None
        self.logger = None


    def reset_tape(self, trainables):
        self.grad_tape = tf.GradientTape(persistent=True)
        self.grad_tape._push_tape()
        self.grad_tape.watch(trainables)

        return self.grad_tape

    def apply_tape(self, grad_tape, loss, trainables):

        self.grad_tape = grad_tape

        loss = tf.cast(loss, tf.float32)

        if self.dynamic_grad_alpha:
            self.grad_alpha = self.grad_alpha * self.one_minus_dga_rate + (1 / tf.reduce_max(loss)) * self.dga_rate

        loss = loss*self.grad_alpha

        if self.tanh_grad_aplha:
            loss = tf.math.tanh(loss)

        self.grad_tape._pop_tape()

        grads = self.grad_tape.gradient(loss, trainables)
        self.optimizer.apply_gradients(zip(grads, trainables))

        if not self.logger is None and not self.name is None:
            self.logger.log_mean(str(self.name) + '_loss', np.mean(loss.numpy()))

        return loss
