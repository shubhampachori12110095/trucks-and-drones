import numpy as np
import tensorflow as tf

class LoggingTensorArray:

    def __init__(self, name, logger):

        self.name = name
        self.logger = logger

    def reset_tensor(
            self,
            dtype,
            size= None,
            dynamic_size= None,
            clear_after_read= None,
            tensor_array_name= None,
            handle= None,
            flow= None,
            infer_shape= True,
            element_shape= None,
            colocate_with_first_write_call= True,
            op_name= None
    ):

        self.tf_array = tf.TensorArray(
            dtype, size, dynamic_size, clear_after_read, tensor_array_name, handle, flow, infer_shape, element_shape,
            colocate_with_first_write_call, op_name)

    def write_to_tensor(self,t, val):

        self.tf_array = self.tf_array.write(index=t,value=val)

    def stack_tensor(self, op_name=None, exp_dims=True, log_mean=True, log_sum=False):

        tf_array = tf.squeeze(self.tf_array.stack(op_name))

        if log_mean:
            self.logger.log_mean(str(self.name), np.mean(tf_array.numpy()))

        if log_sum:
            self.logger.log_mean(str(self.name)+'_sum', np.sum(tf_array.numpy()))

        if exp_dims:
            return tf.expand_dims(tf_array, 1)
        return tf_array

