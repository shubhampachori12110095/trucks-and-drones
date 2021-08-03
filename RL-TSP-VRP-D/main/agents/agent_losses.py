import tensorflow as tf


def discrete_actor_loss(act_probs, advantage, loss_critic):
    act_log_probs = tf.math.log(act_probs)
    loss_actor = -tf.math.reduce_sum(act_log_probs * advantage)
    return loss_actor