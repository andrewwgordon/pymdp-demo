""" Quickly build a random active inference agent, give it an observation and have it do hidden state and policy inference """

import pymdp
from pymdp import utils
from pymdp.agent import Agent

num_obs = [3, 5] # observation modality dimensions
num_states = [4, 2, 3] # hidden state factor dimensions
num_controls = [4, 1, 1] # control state factor dimensions
A_array = utils.random_A_matrix(num_obs, num_states) # create sensory likelihood (A matrix)
B_array = utils.random_B_matrix(num_states, num_controls) # create transition likelihood (B matrix)

C_vector = utils.obj_array_uniform(num_obs) # uniform preferences

# instantiate a quick agent using your A, B and C arrays
my_agent = Agent( A = A_array, B = B_array, C = C_vector)

# give the agent a random observation and get the optimized posterior beliefs

observation = [1, 4] # a list specifying the indices of the observation, for each observation modality

qs = my_agent.infer_states(observation) # get posterior over hidden states (a multi-factor belief)

# Do active inference

q_pi, neg_efe = my_agent.infer_policies() # return the policy posterior and return (negative) expected free energies of each policy as well

action = my_agent.sample_action() # sample an action from the posterior over policies

print(observation)
print(qs)
print(action)