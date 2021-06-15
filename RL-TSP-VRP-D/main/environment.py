import gym
from gym import spaces

from main.simulation.simulation import BaseSimulator
from main.simulation.action_interpreter import BaseActionInterpreter
from main.simulation.state_interpreter import BaseStateInterpreter
from main.reward_calculator import BaseRewardCalculator
from main.visualizer import BaseVisualizer

#from logger import TrainingLogger, TestingLogger

'''
überbegriff für travelling salesman und vehicle routing problem
'''
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {
    'render.modes': ['human'],
    }

    def __init__(self, 
            name,
            all_parameter_list,
            simulation_param,
            visual_param,
            output_param,
            input_param,
            reward_param,
            #render_mode = ???,
            ):

        super(CustomEnv, self).__init__()

        # Init simulator
        self.simulator     = BaseSimulator(all_parameter_list, simulation_param)

        # Init visulizor:
        self.visualizor    = BaseVisualizer(name, visual_param, self.simulator)
        
        # Init state and action interpreter
        self.action_interp = BaseActionInterpreter(output_param, self.simulator, only_at_node_interactions=False)
        self.state_interp  = BaseStateInterpreter(input_param, self.visualizor, self.simulator, img_inputs=False)
        
        # Init reward calculator
        self.reward_calc   = BaseRewardCalculator(reward_param, self.simulator)

        # Init Logger (move to train process)
        #self.logger        = TrainingLogger()
        #self.test_logger   = TestingLogger()

        # Init gym spaces:
        self.reset()
        self.action_space      = self.action_interp.action_space()
        self.observation_space = self.state_interp.obs_space()

        # Init Counter:
        self.count_episodes    = 0
        self.count_total_steps = 0

    def step(self, actions):
        
        # take action:
        self.simulator.temp_db.init_step()
        self.action_interp.take_actions(actions)
        self.simulator.temp_db.finish_step()

        # new state:
        observation, done = self.state_interp.observe_state()

        # reward:
        reward = self.reward_calc.reward_function()

        self.count_steps_of_episode += 1
        self.count_total_steps      += 1

        if done:
            self.count_episodes     += 1

        return observation, reward, done, {}


    def reset(self):

        # reset counter:
        self.count_steps_of_episode = 0

        self.simulator.reset_simulation()

        # Init first state:
        observation, self.done = self.state_interp.observe_state()

        return observation
        
    def render(self, mode='human', close=False):
        if mode == 'human':
            self.visualizor.visualize_step(self.count_episodes, self.count_steps_of_episode)

        if close == True:
            self.visualizor.close()


