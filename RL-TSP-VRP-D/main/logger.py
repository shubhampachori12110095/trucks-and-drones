import tensorflow as tf

class BaseLogger:

    """
    Creates a summary writer for tensorboard logging.

    Args:
        name (string): Name of the model.
        log_dir (string): Path that indicates which dataset is used.
    """

    def __init__(self, name, log_dir):

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
    def __init__(self):
        super().__init__()


class TestingLogger(BaseLogger):
    def __init__(self):
        super().__init__()

