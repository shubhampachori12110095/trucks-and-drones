# Trucks and Drones (RL-TSP-VRP-D)
Using reinforcement learning to solve the travelling salesman and vehicle routing problem with drones (or robots).


## Installation

- Install Trucks and Drones with pip:

```
pip install trucks-and-drones
```

- Install with git:

```
git clone https://github.com/maik97/trucks-and-drones.git
cd trucks-and-drones
python setup.py install
```


## Dependencies

- gym >= 0.17.3


## Documentation 

https://trucks-and-drones.rtfd.io/


## Example

```python
from stable_baselines3 import PPO
from trucks_and_drones import BuildEnvironment

env = BuildEnvironment('test')

env.trucks(1)
env.depots(1)
env.customers(10)

env.observations(contin_inputs = [['v_coord'], ['c_coord'], ['d_coord'], ['demand']])
env.actions(discrete_outputs = ['nodes'])

env.compile()
env = env.build()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200000) #  takes about 10 minutes with cpu


obs = env.reset()
sum_rewards = 0

for i in range(100):

    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    sum_rewards += reward

    env.render(slow_down_pls=True)
    
    if done:
        print('sum_rewards:', sum_rewards)
        sum_rewards = 0
        obs = env.reset()

env.close()
```


## Citing

If you use `trucks-and-drones` in your research, you can cite it as follows:

```bibtex
@misc{schürmann2021trucks-and-drones,
    author = {Maik Schürmann},
    title = {trucks-and-drones},
    year = {2021},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/maik97/trucks-and-drones}},
}
```
