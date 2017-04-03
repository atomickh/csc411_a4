"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *

import pickle
import sys



parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-l', '--load-model', metavar='NPZ',
                    help='NPZ file containing model weights/biases')
args = parser.parse_args()



env = gym.make('CartPole-v0')


RNG_SEED=3
tf.set_random_seed(RNG_SEED)
env.seed(RNG_SEED)

hidden_size = 2
alpha = 1e-3
TINY = 1e-8
gamma = 0.99

weights_init = xavier_initializer(uniform=False)
relu_init = tf.constant_initializer(0.1)

# if args.load_model:
#     model = np.load(args.load_model)
#     hw_init = tf.constant_initializer(model['hidden/weights'])
#     hb_init = tf.constant_initializer(model['hidden/biases'])
#     mw_init = tf.constant_initializer(model['mus/weights'])
#     mb_init = tf.constant_initializer(model['mus/biases'])
#     sw_init = tf.constant_initializer(model['sigmas/weights'])
#     sb_init = tf.constant_initializer(model['sigmas/biases'])
# else:
#     hw_init = weights_init
#     hb_init = relu_init
#     mw_init = weights_init
#     mb_init = relu_init
#     sw_init = weights_init
#     sb_init = relu_init
    
hw_init = weights_init
hb_init = relu_init
mw_init = weights_init
mb_init = relu_init
sw_init = weights_init
sb_init = relu_init
try:
    output_units = env.action_space.shape[0]
except AttributeError:
    output_units = env.action_space.n

output_units = 2

input_shape = env.observation_space.shape[0]
NUM_INPUT_FEATURES = 4
x = tf.placeholder(tf.float32, shape=(None, NUM_INPUT_FEATURES), name='x')
y = tf.placeholder(tf.int32, shape=(None, output_units), name='y')


hidden = fully_connected(
    inputs=x,
    num_outputs=output_units,
    activation_fn=None,
    weights_initializer=hw_init,
    weights_regularizer=None,
    biases_initializer=hb_init,
    scope='hidden')

# mus = fully_connected(
#     inputs=hidden,
#     num_outputs=output_units,
#     activation_fn=tf.tanh,
#     weights_initializer=mw_init,
#     weights_regularizer=None,
#     biases_initializer=mb_init,
#     scope='mus')
# 
# sigmas = tf.clip_by_value(fully_connected(
#     inputs=hidden,
#     num_outputs=output_units,
#     activation_fn=tf.nn.softplus,
#     weights_initializer=sw_init,
#     weights_regularizer=None,
#     biases_initializer=sb_init,
#     scope='sigmas'),
#     TINY, 5)
    

all_vars = tf.global_variables()

out = softmax(hidden, scope="out")
#print(out)

pi = tf.contrib.distributions.Bernoulli(p=out, name='pi')
#pi_sample = tf.argmax(out, 1, name="pi_sample")
pi_sample = pi.sample()
log_pi = pi.log_prob(y, name='log_pi')

Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_op = optimizer.minimize(-1.0 * Returns * log_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEMORY=20
MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

steps = []
ret = []
mean = []
#MAX_STEPS = 5
track_returns = []
for ep in range(6000):
    obs = env.reset()

    G = 0
    ep_states = []
    ep_actions = []
    ep_rewards = [0]
    done = False
    t = 0
    I = 1
    while not done:
        #print(obs)
        ep_states.append(obs)
        env.render()

        action = sess.run([pi_sample], feed_dict={x:[obs]})[0][0]
        ep_actions.append(action)
        obs, reward, done, info = env.step(action[0])
        ep_rewards.append(reward * I)
        G += reward * I
        I *= gamma

        t += 1
        if t >= MAX_STEPS:
            break

    if not args.load_model:

        returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
        index = ep % MEMORY
        
       
        _ = sess.run([train_op],
                    feed_dict={x:np.array(ep_states),
                                y:np.array(ep_actions),
                                Returns:returns })

    track_returns.append(G)
    track_returns = track_returns[-MEMORY:]
    mean_return = np.mean(track_returns)
    print("Episode {} finished after {} steps with return {}".format(ep, t, G))
    
    print("Mean return over the last {} episodes is {}".format(MEMORY,
                                                               mean_return))
    
    ret.append(G)
    steps.append(t)
    mean.append(mean_return)

    #with tf.variable_scope("mus", reuse=True):
        #print("incoming weights for the mu's from the first hidden unit:", sess.run(tf.get_variable("weights"))[0,:])

save = True
if save:
    snapshot = {}
    snapshot["steps"] = steps
    snapshot["ret"] = ret
    snapshot["mean"] = mean
    pickle.dump(snapshot,  open("new_snapshot_steps.pkl", "wb"))
sess.close()
