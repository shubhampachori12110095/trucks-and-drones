.. _getting_started:

Getting Started Guide
=====================

``peak-shaver`` aims to provide the tools to explore different approaches of reinforcement learning within a simulation of the `HIPE Dataset <https://www.energystatusdata.kit.edu/hipe.php>`_ . The module for the simulation ``common_env`` is made as a ``gym`` environment, which provides a common API for a wide range of different RL-libraries (for example ``stable-baseline`` which is also used as part of the study project). You can also create your own Agents following the ``gym`` guide-lines, but note that ``common_env`` can provide some extra functionality (look up the :ref:`module <common_env_doc>`, for example the extra return `episode_max_peak` in :meth:`common_env.common_env.step`). Furthermore the module ``reward_maker`` is used to specify the kind of reward the agent will receive.

Installation and Dependencies
*****************************

You can download the zip file from the `github repository <https://github.com/maik97/peak-shaver>`_ (alternatively just clone the project to your own github) or run the command below if you have `git <https://git-scm.com/downloads>`_ installed.

.. code-block:: console
   
    $ git clone git://github.com/maik97/peak-shaver.git

Make sure to have these libraries with the right versions installed (espacially tensorflow and tensorboard, the other libraries should be possible with >=[version]):

- numpy==1.19.4
- pandas==0.25.3
- matplotlib==3.3.3
- seaborn==0.11.0
- keras==2.0.8
- tensorflow==1.9.0
- gym==0.17.3
- stable-baselines==2.10.1
- h5py==2.10.0

If you dont know how to install those properly look up `pip <https://pip.pypa.io/en/stable/>`_ . You can also install all dependedencies at once via the requirements.txt found in the github repository.

Note that ``tensorflow 1.9.0`` is an older version and only works with ``python 3.6``. The code of ``logger`` needs to be updated in order to be compatible with of ``tensorflow 2.x.x``. (This can't be guaranteed though)

The dataset can be downloaded here: `HIPE Dataset <https://www.energystatusdata.kit.edu/hipe.php>`_ . There are two different versions, one is the complete dataset over three months, the smaller one is just the first week.

Folder Structure
****************

| peak-shaver-master
| ├── peak-shaver
| │   ├── dataset
| │   │   ├── hipe_cleaned_v1.0.1_geq_2017-10-23_lt_2017-10-30
| │   │   └── hipe_cleaned_v1.0.1_geq_2017-10-01_lt_2018-01-01
| │   ├── (_BIG_D)
| │   ├── (_small_d)
| │   ├── [Put here any of your own code]
| │   └── ...
| └── ...

- ``peak-shaver-master`` is the downloded github folder.
- ``peak-shaver`` is where the actual package is located. When following the examples or if you want to create your own code you should be in this directory.
- ``dataset``: put in (both) unzipped HIPE-datasets.
- ``_BIG_D`` (big dataset) and ``_small_d`` (small dataset): this is where datasets, models, statistics and logs will be saved. Note that those folders will be created by setting the parameter `D_PATH` and therfore can be named differently. More on this in the next section.

Data Preparation
****************
The data preparation will be executed automaticaly when you first run ``wahrsager`` or any of the agents (provided you didn't do it manually). But it is recommended to create the preparations separately with ``schaffer`` since this can take up some time. If you decide to create all the datasets at once you can use `peak-shaver-master/peak-shaver/common_settings.py`. This can also provide a standart setup for the agents, so you don't have to create a setup for all agents manually.

Create the basic dataset:

.. code-block:: python
    
    from main.schaffer import mainDataset, lstmInputDataset
    from main.common_func import wait_to_continue

    # Setup main dataset creater/loader:
    main_dataset = mainDataset(
        D_PATH='_BIG_D/',
        period_min='5',
        full_dataset=True)

    # Run this first, since this can take up a lot of time:
    main_dataset_creator.smoothed_df()
    # wait_to_continue() # Pauses the execution until you press enter

    # These don't take up a lot of time to run, 
    # but you can run those beforhand to check if everything is setup properly:
    main_dataset_creator.load_total_power()
    main_dataset_creator.normalized_df()
    main_dataset_creator.norm_activation_time_df()
    # wait_to_continue()

- :meth:`schaffer.mainDataset.smoothed_df` will take the dataset and smooth the data to a specific time-frame.
- :meth:`schaffer.mainDataset.load_total_power` will take the table from ``smoothed_df`` and calculates the (not normalized) sum of the power requirements.
- :meth:`schaffer.mainDataset.normalized_df` will take the table from ``smoothed_df`` and normalize the data
- :meth:`schaffer.mainDataset.norm_activation_time_df` will take the table from ``smoothed_df`` and calculate the normalized activation times of the machines.

In this tutorial we seperate the big and small datasets, by setting ``D_PATH=_BIG_D`` for the big one and ``D_PATH=_BIG_D`` for the small one. Dont forget to set ``full_dataset=False`` if you want to use the small dataset. ``period_min`` can be set to an integer that defines the minutes of one period. :meth:`common_func.wait_to_continue` pauses the code, so you have time to check out the created datasets.

Create an input-dataset:

.. code-block:: python
    
    # Continuation from the code above (needs `main_dataset` and imports)

    # Import main dataset as dataframe:
    df = main_dataset.make_input_df(
        drop_main_terminal=False,
        use_time_diff=True,
        day_diff='holiday-weekend')

    # Setup lstm dataset creator/loader:
    lstm_dataset = lstmInputDataset(main_dataset, df, num_past_periods=12)

    # If you want to check that everything works fine, run those rather step by step:
    lstm_dataset_creator.rolling_mean_training_data()
    #wait_to_continue()

    lstm_dataset_creator.rolling_max_training_data()
    #wait_to_continue()

    lstm_dataset_creator.normal_training_data()
    #wait_to_continue()

    lstm_dataset_creator.sequence_training_data(num_seq_periods=12)
    #wait_to_continue()

- :meth:`schaffer.lstmInputDataset.rolling_mean_training_data` creates an input-dataset that was transformed with a `rolling mean` operation
- :meth:`schaffer.lstmInputDataset.rolling_max_training_data` creates an input-dataset that was transformed with a `rolling max` operation
- :meth:`schaffer.lstmInputDataset.normal_training_data` creates a normale input-dataset.
- :meth:`schaffer.lstmInputDataset.normal_training_data` creates an input-dataset with sequence-labels the size of ``num_seq_periods``.


Making Predictions
******************
Following the same principle above (time consumption, more freedom to set up) it is also recommended to make the predictions seperately, although this will also be done automatically provided you didn't do it manually. 

With the module ``wahrsager`` you can train an LSTM that aims to predict the future power consumption. It's possible to modify the ``main`` function and run ``wahrsager`` directly. You can also create your own python code following this example:

.. code-block:: python
    
    ''' Example code to train a LSTM using the wahrsager module'''
    from main.wahrsager import wahrsager
    from main.common_func import max_seq, mean_seq

    # Predictions (and training) with different approaches:
    prediction_mean           = wahrsager(PLOTTING=True, TYPE='MEAN').train()
    prediction_max            = wahrsager(PLOTTING=True, TYPE='MAX').train()
    prediction_normal         = wahrsager(PLOTTING=True, TYPE='NORMAL').train()
    prediction_max_label_seq  = wahrsager(PLOTTING=True, TYPE='MAX_LABEL_SEQ').train()
    prediction_mean_label_seq = wahrsager(PLOTTING=True, TYPE='MEAN_LABEL_SEQ').train()

    prediction_seq      = wahrsager(PLOTTING=True, TYPE='SEQ', num_outputs=12).train()
    max_prediction_seq  = max_seq(prediction_seq)
    mean_prediction_seq = mean_seq(prediction_seq)

:meth:`wahrsager.wahrsager.train()` function is used to train a LSTM-model and will return predictions after the training is complete. You can use :meth:`wahrsager.wahrsager.pred()` once you have run the training for the first time (will be used by the agents). You can find the saved models in either _BIG_D/LSTM-models/ or _small_d/LSTM-models/.

There are different approaches to modify the input-dataset, which can be set with ``TYPE=...``. Below are explanations of the variables from the code snippet which are returns from a LSTM with a different ``TYPE``.

- ``prediction_mean`` with ``TYPE='MEAN'``: Predictions of the dataset modified with a rolling mean
- ``prediction_max`` with ``TYPE='MAX'``: Predictions of the dataset modified with a rolling max
- ``prediction_normal`` with ``TYPE='NORMAL'``: Predictions of the unmodified dataset
- ``prediction_max_label_seq`` with ``TYPE='MAX_LABEL_SEQ'``: Predictions where just the label data is modified with a rolling max
- ``prediction_mean_label_seq`` with ``TYPE='MEAN_LABEL_SEQ'``: Predictions where just the label data is modified with a rolling mean
- ``prediction_seq`` with ``TYPE='SEQ'``: Sequence-Predictions of the unmodified dataset, each sequence can be transformed to the mean or max value with ``max_seq(prediction_seq)`` or ``mean_seq(prediction_seq)``

All these different approaches will have similar results, but can be used to optimize the predictions. If you want to tune the parameters, look up the ``wahrsager`` class :ref:`here <wahrsager_doc>` (change time-frame, LSTM size, ...). Note that for every new time-frame a separate dataset will be created.

Set ``PLOTTING=True`` if you want to see a graph of the predictions compared to the actual data. You also can find the saved graphs in either _BIG_D/LSTM-graphs/ or _small_d/LSTM-graphs/. An example graph is provided below:

- hier kommt beispiel graph