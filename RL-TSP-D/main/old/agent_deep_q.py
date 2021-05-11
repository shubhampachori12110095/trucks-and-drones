import numpy as np
import random

from collections import deque

from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

try:
    from main.common_func import try_training_on_gpu, AgentStatus
except:
    from common_func import try_training_on_gpu, AgentStatus

class DQN:
    """
    Deep Q Network
    
    Args:
        env (object): Takes in a ``gym`` environment, use the common_env to simulate the HIPE-Dataset.
        memory_len (int): Sets the memory length of a deque object. The deque object is used to store past steps, so they can be learned all at once. Use this based on your update number and possible on the multi-step-, lstm-hotizon.
        input_sequence (int): Length of the input steps, must be used when the neural network has an LSTM
        gamma (float): Factor that determines the importance of futue Q-values, value between 0 and 1
        epsilon (float): Initial percent of random actions, value between 0 and 1
        epsilon_min (float): Minimal percent of random actions, value between 0 and 1
        epsilon_decay (float, string): Factor by which ``epsilon`` decays, value between 0 and 1. Can also be set to `linear` when you want epsilon to decrease over all steps (In this case ``epsilon_min`` will not be used).
        lr (float): Sets the learning rate of the RL-Agent
        tau (float): Factor for copying weights from model network to target network
        activation (string): Defines Keras activation function for each Dense layer (except the ouput layer) for the DQN
        loss (string): Defines Keras loss function to comile the DQN model
        model_type (string): Use `dense` to use an normal neural network and `lstm` to integrate an LSTM to the neural network.
        use_model (Kreras model): Can be used to pass a pre-defined Keras model instead. When you use this check if the input and output shapes are correct, since those wont be dynamic in this case.
        pre_trained_model (string): Name of an existing Keras model in `[D_PATH]/agent-models`. Loads a pre-trained model (Note that the table must be in .h5 format).
        hidden_layers (int): Number of hidden layers for the neural network
        hidden_size (int): Size of all hidden layers
        lstm_size (int): Number of LSTM layers, when ``model_type=lstm``

    """
    def __init__(self, env, memory_len, input_sequence=1, gamma=0.85, epsilon=0.8, epsilon_min=0.1, epsilon_decay=0.999996, lr=0.5, tau=0.125,
                 activation='relu', loss='mean_squared_error', model_type='dense', use_model=None, pre_trained_model=None,
                 hidden_layers=2, hidden_size=518, lstm_size=128, target_update_num=None):

        self.env            = env
        
        self.D_PATH         = self.env.__dict__['D_PATH']
        self.NAME           = self.env.__dict__['NAME']
        self.model_type     = model_type

        self.input_sequence = input_sequence
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay # string:'linear' or float
        self.epsilon_og     = epsilon
        self.tau            = tau

        self.target_update_num = target_update_num

        # Check horizon lenght and create memory deque object:
        self.horizon = self.env.__dict__['reward_maker'].__dict__['R_HORIZON']
        if isinstance(self.horizon, int) == True:
            memory_len += self.horizon
        else:
            self.horizon = 0
        self.memory = deque(maxlen=memory_len+self.input_sequence-1)
        
        # Check which model to use:
        if pre_trained_model == None:
            if model_type == 'dense' and use_model == None:
                self.model        = self.create_normal_model(lr, activation, loss, hidden_layers, hidden_size)
                self.target_model = self.create_normal_model(lr, activation, loss, hidden_layers, hidden_size)

            elif model_type == 'lstm' and use_model == None:
                if self.input_sequence <= 1:
                    raise Exception("input_sequence was set to {} but must be > 1, when model_type='lstm'".format(self.input_sequence))
                self.model        = self.create_lstm_model(lr, activation, loss, lstm_size, hidden_size)
                self.target_model = self.create_lstm_model(lr, activation, loss, lstm_size, hidden_size)

            elif use_model != None:
                self.model        = use_model
                self.target_model = clone_model(self.model)
                self.model_type   = 'costum'
            else:
                raise Exception("model_type='"+model_type+"' not understood. Use 'dense', 'lstm' or pass your own compiled keras model with own_model.")
        else:
            self.model = load_model(self.D_PATH+'agent-models/'+pre_trained_model)

        # Pass logger object:
        self.LOGGER = self.env.__dict__['LOGGER']

        # Check GPU:
        try_training_on_gpu()

        # Check action type:
        if self.env.__dict__['ACTION_TYPE'] != 'discrete':
            raise Exception("DQN-Agent can not use continious values for the actions. Change 'ACTION_TYPE' to 'discrete'!")

        # Check input sequence:
        if self.input_sequence == 0:
            print("Attention: input_sequence can not be set to 0, changing now to input_sequence=1")
            self.input_sequence = 1 


    def init_agent_status(self, epochs, epoch_len):
        '''
        Ininitiates a helper class to make agent status prints

        Args:
            epochs (int): Number of epochs
            epoch_len (int): Number of steps per epoch
        '''
        self.agent_status = AgentStatus(epochs*epoch_len)


    def create_normal_model(self, lr, activation, loss, hidden_layers, hidden_size):
        '''Class-Function: Creates a Deep Neural Network which predicts Q-Values, when the class is initilized.

        Args:
            activation (string): Previously defined Keras activation
            loss (string): Previously defined Keras loss
            hidden_layers (int): Number of hidden layers for the neural network
            hidden_size (int): Size of all hidden layers
        '''
        input_dim = self.env.observation_space.shape[0]*self.input_sequence
        model = Sequential()
        model.add(Dense(hidden_size, input_dim=input_dim, activation=activation))
        model.add(Dense(hidden_size, activation=activation))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss=loss, optimizer=Adam(lr=lr))
        return model


    def create_lstm_model(self, lr, activation, loss, lstm_size, hidden_size):
        '''Class-Function: Creates an LSTM which predicts Q-Values, when the class is initilized.

        Args:
            activation (string): Previously defined Keras activation
            loss (string): Previously defined Keras loss
            hidden_size (int): Size of all hidden layers
            lstm_size (int): Number of LSTM layers, when ``model_type=lstm``
        '''
        input_dim = (self.input_sequence, self.env.observation_space.shape[0])
        print(input_dim)
        model = Sequential()
        model.add(LSTM(lstm_size, input_shape=(self.input_sequence, self.env.observation_space.shape[0])))
        model.add(Dense(hidden_size, activation=activation))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss=loss, optimizer=Adam(lr=lr))
        return model


    def sequence_states_to_act(self, cur_state):
        '''
        Class-Function: Uses the memory to create a input sequence of multiple steps for :meth:`agent_deep_q.act` (For example used when ``model_type=lstm``)

        Args:
            cur_state (array): State from current step

        Returns:
            array: Multiple past states
        '''
        # Make multi-state-input array:
        state_array = []
        for i in range(self.input_sequence-1):
            state, action, reward, new_state, done, step_counter_episode = self.memory[-self.input_sequence+1+i]
            state_array = np.append(state_array, state)
        state_array = np.append(state_array, cur_state)

        # Reshape array that can be passed to an LSTM input:
        if self.model_type == 'lstm':
            state_array = state_array.reshape(-1,self.input_sequence,len(cur_state[0]))

        return state_array


    def act(self, state, random_mode=False, test_mode=False):
        '''
        Function, in which the agent decides an action, either from greedy-policy or from prediction. Use this function when iterating through each step.
        
        Args:
            state (array): Current state at the step

        Returns:
            integer: Action that was chosen by the agent
        '''
        # Calculate new epsilon:
        if self.epsilon_decay == 'linear':
            self.epsilon = 1-(self.epsilon_og*(self.env.__dict__['sum_steps']/self.agent_status.__dict__['max_steps']))
        else:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        
        # Random decision if action will be random:
        if np.random.random() < self.epsilon or random_mode == True:
            if test_mode == False:
                return self.env.action_space.sample()
        
        # If action is not random:
        # Discrete states as input:
        if self.env.__dict__['num_discrete_obs'] > 1:
            shape = self.env.__dict__['num_discrete_obs']
            state = self.discrete_input_space(shape,state)
        # Continious state as input:
        else:
            state = state.reshape(1,len(state))

        # Decide if ´multiple or single state as input:
        if self.input_sequence > 1:
            state = self.sequence_states_to_act(state)
        return np.argmax(self.model.predict(state)[0])


    def remember(self, state, action, reward, new_state, done, step_counter_episode):
        '''
        Takes in all necessery variables for the learning process and appends those to the memory. Use this function when iterating through each step.

        Args:
            state (array): State at which the agent chose the action
            action (integer): Chosen action
            reward (float): Real reward for the action
            new_state (array): State after action is performed
            done (bool): If True the episode ends
            step_counter_episode (integer): Episode step at which the action was performed
        '''
        # Discrete Input:
        if self.env.__dict__['num_discrete_obs'] > 1:
            shape     = self.env.__dict__['num_discrete_obs']
            state     = self.discrete_input_space(shape,state)
            new_state = self.discrete_input_space(shape,state)
        # Continious input:
        else:
            state     = state.reshape(1,len(state))
            new_state = new_state.reshape(1,len(new_state))
        self.memory.append([state, action, reward, new_state, done, step_counter_episode])


    def sequence_states_for_replay(self,i):
        '''
        Class-Function: Uses the memory to create a input sequence of multiple steps for :meth:`agent_deep_q.replay` (For example used when ``model_type=lstm``).
        Note that this function is still inefficient and makes the learning process for the lstm pretty slow.


        Args:
            i (int): Current itaration through the memory

        Returns:
            tuple (state_array, action, reward, new_state_array, done, step_counter_episode):

            state_array (array): Multiple past states
            
            action (integer): Chosen action
            
            reward (float): Real reward for the action
            
            new_state (array): State after action is performed
            
            done (bool): If True the episode ended
            
            step_counter_episode (integer): Episode step at which the action was performed
        '''
        # Make multi-state-input array:
        state_array     = []
        new_state_array = []
        for i in range(self.input_sequence):
            state, action, reward, new_state, done, step_counter_episode = self.memory[-self.input_sequence+1+i]
            state_array     = np.append(state_array, state)
            new_state_array = np.append(new_state_array, new_state)
        
        # Reshape array that can be passed to an LSTM input:
        if self.model_type == 'lstm':
            state_array     = state_array.reshape(-1,self.input_sequence,len(state[0]))
            new_state_array = new_state_array.reshape(-1,self.input_sequence,len(state[0]))
        
        return state_array, action, reward, new_state_array, done, step_counter_episode


    def replay(self, batch_size):
        '''
        Training-Process for the DQN from past steps. Use this function after a few iteration-steps (best use is the update_num, alternatively use this function at each step).

        Args:
            batch_size (integer): Number of past states to learn from
        '''
        # Create training batch data:
        state_batch  = []
        target_batch = []
        for i in range(batch_size):
            # Load single state:
            if self.input_sequence == 1:
                state, action, reward, new_state, done, step_counter_episode = self.memory[i]
            # Load multi state:
            else:
                state, action, reward, new_state, done, step_counter_episode = self.sequence_states_for_replay(i)

            target = self.target_model.predict(state)
            
            # Check for multi-step rewards:
            if reward == None:
                reward = self.env.get_multi_step_reward(step_counter_episode)

            # Check if epoch is finished:
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma

            state_batch  = np.append(state_batch, state)
            target_batch = np.append(target_batch, target)


        # Reshaping
        if self.model_type != 'lstm':
            state_batch  = state_batch.reshape(-1,len(state[0]))
            target_batch = target_batch.reshape(-1,len(target[0]))
        else:
            state_batch  = state_batch.reshape(batch_size, len(state[0]),-1)
            target_batch = target_batch.reshape(batch_size, len(target[0]))

        # Prepare agent status to log and print:
        name         = self.env.__dict__['NAME']
        epoch        = self.env.__dict__['episode_counter']
        total_steps  = self.env.__dict__['sum_steps']
        
        # Training:
        loss         = self.model.train_on_batch(state_batch,target_batch)
        #self.target_train()
        # Print status
        self.agent_status.print_agent_status(name, epoch, total_steps, batch_size, self.epsilon, loss) 
        
        # Log status:
        self.LOGGER.log_scalar('Loss:', loss, total_steps, done)
        self.LOGGER.log_scalar('Epsilon:', self.epsilon, total_steps, done)


    def target_train(self):
        '''Updates the Target-Weights. Use this function after :meth:`agent_deep_q.replay`'''
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)


    def save_agent(self, e):
        '''For saving the agents model at specific epoch. Make sure to not use this function at each epoch, since this will take up your memory space.

        Args:
            e (integer): Takes the epoch-number in which the model is saved
        '''
        self.model.save(self.D_PATH+'agent-models/'+self.model_type+self.NAME+'_{}'.format(e))


    def discrete_input_space(state_dim_size, state):
        '''
        Transforms the state to a discrete input (one-hot-encoding)

        Args:
            state_dim_size (shape): Size to which the state will be transformed
            state (array): Takes in a state

        Returns:
            discrete_state (array): Transformed state for discrete inputs
        '''
        discrete_state = []
        for dim in state:
            dim_append = np.zeros((state_dim_size))
            dim_append[dim] = 1
            discrete_state = np.append(discrete_state,dim_append)
        discrete_state = discrete_state.reshape(1,len(discrete_state))
        #print(discrete_state)
        return discrete_state
