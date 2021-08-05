import tensorflow as tf
import numpy as np

class BaseLogger:

    """
    Creates a summary writer for tensorboard logging.

    Args:
        name (string): Name of the model.
        log_dir (string): Path that indicates which dataset is used.
    """

    def __init__(self, name, log_dir):
        self.name = name

        #logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.file_writer = tf.summary.create_file_writer(log_dir+"/"+name)
        self.file_writer.set_as_default()


    def log_scalar(self, tag, value, step):
        """Log a scalar variable.

        Args:
            tag (string): Name of the scalar
            value (float): Value of the scalar
            step (int): Training iteration
        """
        tf.summary.scalar(tag, value, step=step)
        self.file_writer.flush()


class TrainingLogger(BaseLogger):
    def __init__(self, name, log_dir):
        super().__init__(name, log_dir)

        self.mean_per_episode_dict = {}

    def log_mean(self, tag, value):

        if tag in set(self.mean_per_episode_dict.keys()):
            self.mean_per_episode_dict[tag].append(value)
        else:
            self.mean_per_episode_dict[tag] = [value]


    def print_status(self, episode):

        print('\nname:', self.name)
        print('Episode:', str(episode))
        for key in self.mean_per_episode_dict.keys():
            if len(self.mean_per_episode_dict[key]) > 1:
                self.mean_per_episode_dict[key] = np.mean(self.mean_per_episode_dict[key])
            else:
                self.mean_per_episode_dict[key] = float(np.squeeze(self.mean_per_episode_dict[key]))
            print('{}: {}'.format(key, self.mean_per_episode_dict[key]))
            self.log_scalar(key, self.mean_per_episode_dict[key], episode)

        self.mean_per_episode_dict.clear()

class StatusPrinter:

    def __init__(self, name):

        self.name = name

        self.mean_per_episode_dict = {}

    def log_mean(self, tag, value):

            if tag in set(self.mean_per_episode_dict.keys()):
                self.mean_per_episode_dict[tag].append(value)
            else:
                self.mean_per_episode_dict[tag] = [value]

    def print_status(self, episode):

        print('\nname:', self.name)
        print('Episode:', str(episode))
        for key in self.mean_per_episode_dict.keys():
            if len(self.mean_per_episode_dict[key]) > 1:
                self.mean_per_episode_dict[key] = np.mean(self.mean_per_episode_dict[key])
            else:
                self.mean_per_episode_dict[key] = float(np.squeeze(self.mean_per_episode_dict[key]))
            print('{}: {}'.format(key, self.mean_per_episode_dict[key]))

        self.mean_per_episode_dict.clear()


class TestingLogger(BaseLogger):
    def __init__(self, name, log_dir):
        super().__init__(name, log_dir)

