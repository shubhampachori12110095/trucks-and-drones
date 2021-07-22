'''
Environment TSP and VRP wit drones or robots
'''
import gym
from datetime import datetime


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {
    'render.modes': ['human'],
    }

    def __init__(
            self,
            name,
            simulation,
            visualizer,
            obs_encoder,
            act_decoder,
            reward_calc,
    ):

        super(CustomEnv, self).__init__()

        self.name = name+datetime.now().strftime("_%d-%m-%Y_%H-%M-%S")

        # Init simulator
        self.simulation = simulation

        # Init visualizer:
        self.visualizer = visualizer
        
        # Init state and action interpreter
        self.act_decoder = act_decoder
        self.obs_encoder = obs_encoder
        
        # Init reward calculator
        self.reward_calc = reward_calc

        # Init Counter:
        self.count_episodes    = 0
        self.count_total_steps = 0

        # Init gym spaces:
        self.reset()
        self.act_decoder.finish_init()

        self.action_space      = self.act_decoder.action_space()
        self.observation_space = self.obs_encoder.obs_space()

    def step(self, actions):
        
        # take action:
        self.simulation.temp_db.init_step()
        self.act_decoder.decode_actions(actions)
        done = self.simulation.finish_step()
        self.simulation.temp_db.finish_step()

        # new state:
        observation = self.obs_encoder.observe_state()

        # reward:
        #reward = self.reward_calc.reward_function()
        #reward = -self.simulation.temp_db.total_time_delta()
        reward = self.simulation.temp_db.bestrafung
        #reward = -self.simulation.temp_db.total_time_delta() + self.simulation.temp_db.bestrafung

        self.count_steps_of_episode += 1
        self.count_total_steps      += 1

        info_dict = {
            'possible_nodes': self.simulation.temp_db.possible_nodes(),
        }

        return observation, reward, done, info_dict

    def reset(self):

        # reset counter:
        self.count_steps_of_episode = 0
        self.count_episodes += 1

        self.simulation.reset_simulation()

        # Init first state:
        observation = self.obs_encoder.observe_state()

        return observation
        
    def render(self, mode='human', close=False):
        if mode == 'human':
            self.visualizer.visualize_step(self.count_episodes, self.count_steps_of_episode)

        if close == True:
            self.visualizer.close()
