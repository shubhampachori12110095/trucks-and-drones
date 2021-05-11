"""
Logs scalars to tensorboard without tensor ops.
Modified Code from Michael Gygli: https://git.io/JLN0e
"""

import tensorflow as tf
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np


class Logger(object):
    """
    Creates a summary writer for tensorboard logging.

    Args:
        NAME (string): Name of the model
        D_PATH (string): Path that indicates which dataset is used. Use `_BIG_D/` for the full dataset and `_small_d/` for the small dataset, if you followed the propesed folder structure.
        only_per_episode (bool): If `True` logs will only be saved once per episode instead of every step. This can save space since the log files can get pretty big for training processes with millions of steps.
    """

    def __init__(self, NAME, D_PATH, only_per_episode=False):

        self.NAME             = NAME
        self.D_PATH           = D_PATH
        self.only_per_episode = only_per_episode
        self.dict_scalars     = {}

        log_dir = D_PATH+'agent-logs/'+NAME
        
        try:
            self.writer = tf.summary.FileWriter(log_dir)
        except Exception as e:
            print(e)
            self.writer = tf.summary.create_file_writer(log_dir)


    def add_to_dict(self, tag, value):
        '''
        Triggers each step in :meth:`logger.Logger.log_scalar` when ``only_per_episode=True``. Temporarily stores the value from the step in a dictionary.
        
        Args:
            tag (string): Name of the scalar
            value (float): Value of the scalar
        '''
        if tag in self.dict_scalars:
            self.dict_scalars[tag] = np.append(self.dict_scalars[tag], value)
        else:
            self.dict_scalars[tag] = [value]


    def get_from_dict(self, tag):
        '''
        Triggers at the end of an episode in :meth:`logger.Logger.log_scalar` when ``only_per_episode=True``. Reads the dictionary and calculates the mean before deleting the values.
        Args:
            tag (string): Name of the scalar

        Returns:
            float: Mean scalar-value of one episode
        '''
        value = np.mean(self.dict_scalars[tag])
        self.dict_scalars[tag] = []
        return value


    def log_scalar(self, tag, value, step, done):
        """Log a scalar variable.

        Args:
            tag (string): Name of the scalar
            value (float): Value of the scalar
            step (int): Training iteration
            done (bool): Indicates the end of an episode
        """
        if self.only_per_episode == True:
            self.add_to_dict(tag, value)
            if done == True:
                value = self.get_from_dict(tag)

        if self.only_per_episode == False or done == True:
            try:
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)
            except:
                summary = tf.summary.scalar(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)