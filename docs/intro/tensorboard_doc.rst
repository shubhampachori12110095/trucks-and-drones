.. _tensorboard_doc:

Tensorboard
===========

Make sure to be in the ``peak-shaver`` directory and open up a new command window. If you are using the big dataset type this command to start up tensorboard:

.. code-block:: console
   
    $ tensorboard --logdir=_BIG_D/[log-type]/

For the small dataset:

.. code-block:: console
   
    $ tensorboard --logdir=_small_d/[log-type]/

The ``log-type`` can either be `agent_logs` or `lstm-logs`