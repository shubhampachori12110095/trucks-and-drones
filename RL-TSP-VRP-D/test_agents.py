import gym

from main.agents.build_agent import BaseAgentBuilder

def test_discrete_actor_critic():
    from main.agents.discrete_actor_critic import DiscreteActorCriticCore

    agent = BaseAgentBuilder(gym.make('CartPole-v0'), log_dir='_logs')
    [agent.assign_agent(DiscreteActorCriticCore(),at_index=i) for i in range(len(agent.action_outputs))]
    agent = agent.build(max_steps_per_episode=20000)
    agent.train_agent(60000)


def test_contin_actor_critic():
    from main.agents.contin_actor_critic import ContinActorCriticCore

    agent = BaseAgentBuilder(gym.make('CartPole-v0'), log_dir='_logs')
    [agent.assign_agent(ContinActorCriticCore(transform_to_discrete=True), at_index=i) for i in range(len(agent.action_outputs))]
    agent = agent.build(max_steps_per_episode=20000)
    agent.train_agent(60000)


def test_dqn():
    from main.agents.DQN import DQNCore

    agent = BaseAgentBuilder(gym.make('CartPole-v0'), log_dir='_logs')
    [agent.assign_agent(DQNCore(), at_index=i) for i in range(len(agent.action_outputs))]
    agent = agent.build(max_steps_per_episode=20000)
    agent.train_agent(60000)


def test_dual_dqn():
    from main.agents.DualDQN import DualDQNCore

    agent = BaseAgentBuilder(gym.make('CartPole-v0'), log_dir='_logs')
    [agent.assign_agent(DualDQNCore(), at_index=i) for i in range(len(agent.action_outputs))]
    agent = agent.build(max_steps_per_episode=20000)
    agent.train_agent(60000)


test_discrete_actor_critic()