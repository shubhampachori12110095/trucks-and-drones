.. RL-TSP-VRP-D documentation master file, created by
   sphinx-quickstart on Thu Jun  3 14:43:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RL-TSP-VRP-D's documentation!
========================================

``RL-TSP-VRP-D`` aims to provide the tools to explore different approaches of reinforcement learning to interact with a simulation of the travelling salesman and vehicle routing problem extended by drones or robots. The module for the simulation ``environment`` is made as a ``gym`` environment, which provides a common API for a wide range of different RL-libraries (for example ``stable-baseline``). You can also create your own Agents following the ``gym`` guidelines and the tutorials of this documentation.

This documentation provides an explanation for the code in collaboration with the bachelor thesis: `Using Reinforcement Learning to solve the Traveling Salesman Problem with Drones <https://www.stackoverflow.com>`_ and the research project `Using Reinforcement Learning to solve the Vehicle Routing Problem with Drones <https://www.stackoverflow.com>`_. Further details can be read here: :ref:`About the project <about_project>`. 

Information about the installation are found in the :ref:`Installation Guide <installation>`. Detailed examples to each RL-agent from the project are provided in the Examples section. In the section Module Documentation, you can find brief explanations for each class and function used in the main modules.

You are free to use this code for your own research (Check this license). If you want to utilize this solution tailored to your company or need any other assistance in implementing artificial intelligence, contact me: maik.schuermann97@gmail.com.



Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Introduction:

   intro/about_project
   intro/installation
   intro/tensorboard_doc

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials/chap_0
   tutorials/chap_1_basic_idea

.. toctree::
   :maxdepth: 2
   :caption: Module Documentations:
   
   modules/logger_doc

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
