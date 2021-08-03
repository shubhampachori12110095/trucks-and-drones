import tensorflow as tf


def discrete_actor_loss(act_probs, advantage, loss_critic, entropy_factor=tf.constant(0.0001, dtype=tf.float32)):

    act_log_probs = tf.math.log(act_probs)
    loss_actor = -(tf.squeeze(act_log_probs) * tf.squeeze(advantage))

    if not entropy_factor is None:
        loss_actor = loss_actor + (entropy_factor * (tf.math.multiply(tf.squeeze(act_probs), tf.squeeze(act_log_probs))))

    return tf.expand_dims(tf.squeeze(loss_actor),1)

def discrete_actor_loss_reduce_mean(act_probs, advantage, loss_critic, entropy_factor=tf.constant(0.0001, dtype=tf.float32)):

    act_log_probs = tf.math.log(act_probs)
    loss_actor = -tf.math.reduce_mean(act_log_probs * advantage)

    if not entropy_factor is None:
        loss_actor = loss_actor + (entropy_factor * tf.reduce_mean(tf.math.multiply(act_probs, act_log_probs)))

    return loss_actor