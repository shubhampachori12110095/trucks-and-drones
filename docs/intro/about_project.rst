.. _about_project:

About the project
=================

With its continuous development and fast improvements artificial intelligence is becoming more and more powerful. One of the big topics in artificial intelligence is Reinforcement Learning (RL), which is useable in any decision-based task (e.g., games). Recently RL algorithms were able to beat humans at the board game GO and at video games like StarCraft or Dota2. This is especially impressive considering the games complexity. With the complexity the number of possibilities and outcomes become too great to be solvable through exact calculations – an optimal solution can’t be calculated with certainty. By considering that the RL algorithm were able to beat the world champions at there respective game, its possible to assume that the algorithms have the potential to reach at least good to almost optimal solutions. 

Other examples of decision-based tasks that can be unsolvable for optimal solutions are the travelling salesman (TSP) and vehicle routing problem (VRP). Both problems are based on satisfying customer demands by visiting each customer to deliver some products. The goal for both problems is to take the most efficient route while visiting each customer. While the TSP uses only one vehicle, the VRP must be solved for multiple vehicles. So, depending on the number of customers and number of vehicles the problems can quickly become unsolvable for exact solutions. With the recent developments in RL, it’s reasonable to assume that RL algorithms can be fitted to solve the TSP and VRP. 

The goal of this project is to implement and test RL against heuristic approaches and calculated exact solutions for TSP and VRP. Both TSP and VRP will also be tested with the extension of using drones or robots. 

Objectives
**********

Implementations:

- Q-Learning: DQN
- Policy Gradients: Actor-Critic, PPO
- Gym Environment: a highly configurable environment to simulate TSP and VRP with possible extension like drones or robots

Research:

- Train on different TSP and VRP tasks
- Test trained RL models against heuristic approaches and exact solutions

Findings
********

- Which RL algorithm was the best choice
- How does it perform against heuristic approaches
- Can it find exact solutions?
- Is RL feasible for solving TSP or VRP?

Future Research
***************

- Other algorithm that could perform better?
- Other neural network architecture?
