.. _chap_1_basic_idea:

Chapter 1 - Introduction and Basic Ideas
========================================

The Tutorials sections aims to provide an overview of the development for the module ``RL-TSP-VRP-D``. Since the project was developed with high flexibility in mind, it can be difficult to comprehend the module in all its details. That’s why the tutorial section aims to explain the details behind the development and at the same time provides a better understanding on how the module can be utilized. The reasoning behind the highly flexible approach is to create the possibility to explore a wide range of reinforcement learning implementations to solve different version of the travelling salesman (TSP) and vehicle routing problem (VRP). Another example for the flexibility is the option to use drones or robots as an extension to TSP and VRP.

With the first chapter the basic ideas and approaches will be presented. At the same time, some first ideas to which extend the module should be flexible will be explored. More details on that will follow in the next chapters.

Vehicles
********

The vehicle routing problem is an extension to the travelling salesman problem. While TSP only uses one vehicle, the VRP must be solved for multiple vehicles. Another extension is the usage of drones or robots, which can be transported by the standard vehicles. The drones or robots can differ to the standard vehicles.  For example, drones can travel by air while standard vehicles and robots can only travel by street. Both drones and robots can also have a limited range (simulated by a battery that needs to be recharged). The following questions explore the options for vehicles:

- How many standard vehicles, drones and robots?
- Can the vehicle be used as a transporter for other vehicles?
- Does the vehicle travel by street or air?
- How fast can the vehicle travel?
- How much cargo (and possibly other vehicles) can be stored in the vehicle?
- Should a transporter store a vehicle in an extra cargo space?
- How much space will a loaded vehicle take up?
- How fast can cargo or other vehicles be unloaded/ loaded?
- Does the vehicle have a max. range?
- Should a possible max. range be simulated by a battery or by a fixed value that simply resets by visiting a transporter or depot?
- Is it possibly to recharge the battery/ reset the max range at both transporters and depots or only at depots?


Nodes
*****

In the vanilla TSP only one depot exists: The vehicle visits each customer node via a given route and returns to the depot node. But to include more options it should also be possible to create more than one depot. Another option might be to make the customer demands more variable. Each customer could have a different demand and perhaps each demand appears at specific time of day. The ideas/ questions for nodes are:

- How many customers and depots?
- How high should demands be and should they be variable?
- When will the demand of a customer appear?
- Does the demand disappear, when not being satisfied after a certain time?
- Can demands recharge?
- Is the stock in a depot unlimited?
- If the stock is limited, can it recharge?


Simulation
**********
The interactions between vehicles and nodes will be simulated. For this purpose, there need to be options to determine which interactions occur at the same time or if there needs to be a specific sequence by prioritisation. Additionally, the grid size needs to be specified.

- What is the grid size?
- (Should transporter always move before being able to unload other vehicles)
- Can transporter unload vehicles and cargo at the same time?
- If same time unloading is possible, what should be prioritized?
- Are there actions that will be executed automatically (e.g., unloading when at customer or loading when at depot)?


Environment
***********
The purpose of the environment is to connect the RL agents with the simulation. Additionaly, two classes will be implemented in order to interpret the actions from the agents and the observations of the simulation. The objects that will be interacting via the environment are:

- Simulator: The simulation of the TSP or VRP (with drones).
- Visualizer: 
- StateInterpreter:
- ActionInterpreter:
- RewardCalculator: