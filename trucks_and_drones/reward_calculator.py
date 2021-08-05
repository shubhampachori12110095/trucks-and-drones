import numpy as np

def reward_parameter(
        reward_modes        = None, #['normalized', 'discounted']
        reward_type         = 'single_vehicle', # 'multi_vehicle', 'sum_vehicle'
        restriction_rewards = ['battery','range','cargo','cargo_rate','cargo_UV','cargo_UV_rate','stock','demand'],
        action_rewards      = ['compare_coord','free_to_travel','unloading_v','free_to_unload_v','free_to_be_loaded_v','free_to_load_v','free_to_unload_cargo','free_to_load_cargo'],
        ):
    return {
        'reward_modes'       : reward_modes,
        'reward_type'        : reward_type,
        'restriction_rewards': restriction_rewards,
        'action_rewards'     : action_rewards,
        }


class RewardFunctions:

    def __init__(self, reward_type, reward_modes, temp_db):

        self.temp_db = temp_db

        if reward_type == 'single_vehicle':
            self.function_list = [self.reward_of_vehicle]
        elif reward_type == 'multi_vehicle':
            self.function_list = [self.rewards_per_vehicle]
        elif reward_type == 'sum_vehicle':
            self.function_list = [self.sum_reward]
        else:
            raise Exception("reward_type was set to {}, but needs to be: 'single_vehicle', 'multi_vehicle' or 'sum_vehicle'")
        
        if isinstance(reward_modes, (list, tuple, np.ndarray)):
            
            if any('normalized' in elem for elem in reward_modes):
                
                ########### ÜBERARBEITEN !!!!!! ############################
                self.max_restrictions = {}
                for elem in self.restriction_rewards:
                    self.max_restrictions[elem] = max([max(abs(self.temp_db.base_groups_restr[elem][i].signal_list)) for i in range(len(self.temp_db.num_vehicles))])
                self.function_list.append(normalize_reward) 
            
            if any('discounted' in elem for elem in reward_modes):
                self.function_list.append(discount_reward) 


    def rewards_per_vehicle(self):

        self.current_reward = [0]**self.temp_db.num_vehicles
        for elem in self.restriction_rewards:
            for i in range(len(self.temp_db.num_vehicles)):
                self.current_reward[i] += self.temp_db.base_groups_restr[elem][i]

        for elem in self.action_rewards:
            for i in range(len(self.temp_db.num_vehicles)):
                self.current_reward[i] += self.temp_db.action_signal[elem][i]

        for i in range(len(self.temp_db.num_vehicles)):
            self.current_reward[i] += self.temp_db.distance_costs[elem][i]


    def reward_of_vehicle(self, vehicle_i):

        self.current_reward = 0
        for elem in self.restriction_rewards:
            self.current_reward[vehicle_i] += self.temp_db.base_groups_restr[elem][vehicle_i]

        for elem in self.action_rewards:
            self.current_reward[vehicle_i] += self.temp_db.action_signal[elem][vehicle_i]

        self.current_reward[vehicle_i] += self.temp_db.distance_costs[elem][vehicle_i]


    def sum_reward(self):

        self.current_reward  = 0
        for elem in self.restriction_rewards:
            for i in range(len(self.temp_db.num_vehicles)):
                self.current_reward[i] += self.temp_db.base_groups_restr[elem][i]

        for elem in self.action_rewards:
            for i in range(len(self.temp_db.num_vehicles)):
                self.current_reward[i] += self.temp_db.action_signal[elem][i]



class BaseRewardCalculator:

    def __init__(self, reward_params, temp_db):
        self.temp_db = temp_db

        # init reward parameter
        [setattr(self, k, v) for k, v in reward_params.items()]

        self.reward_functions = RewardFunctions(self.reward_type, self.reward_modes, temp_db)


    def reward_function(self):
        [reward_func() for reward_func in self.reward_functions.reward_functions]
        print(self.reward_functions.current_reward)
        print(self.temp_db.total_time)
        print(self.reward_functions.current_reward - self.temp_db.total_time)
        return self.reward_functions.current_reward - self.temp_db.total_time