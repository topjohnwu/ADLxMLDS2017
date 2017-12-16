import numpy as np
import gym
import tensorflow as tf
import time
import os
import pickle
from agent_dir.agent import Agent

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# downsampling
def prepro(I):
	""" prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
	I = I[35:195] # crop
	I = I[::2,::2,0] # downsample by factor of 2
	I[I == 144] = 0  # erase background (background type 1)
	I[I == 109] = 0  # erase background (background type 2)
	I[I != 0] = 1    # everything else (paddles, ball) just set to 1
	return I.astype(np.float).ravel()


# tf operations

def tf_discount_rewards(tf_r , gamma): #tf_r ~ [game_steps,1]
	tf_r_reverse = tf.scan(lambda a, v: a * gamma + v, tf.reverse(tf_r, [True, False]))
	tf_discounted_r = tf.reverse(tf_r_reverse, [True, False])
	return tf_discounted_r

def tf_policy_forward(x , tf_model): #x ~ [1,D]
	h = tf.matmul(x, tf_model['W1'])
	h = tf.nn.relu(h)
	logp = tf.matmul(h, tf_model['W2'])
	p = tf.nn.softmax(logp)
	return p

class Agent_PG(Agent):
	def __init__(self, env, args):
		"""
		Initialize every things you need here.
		For example: building your model
		"""

		super(Agent_PG, self).__init__(env)

		tf.reset_default_graph()

		self.n_obs = 80 * 80
		self.hidden_units = 200
		self.n_actions = 3
		self.batch_size = 10
		self.env = env
		self.learning_rate = 1e-3

		self.prev_x = None

		self.tf_model = {}
		with tf.variable_scope('layer_one', reuse=False):
			xavier_l1 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.n_obs), dtype=tf.float32)
			self.tf_model['W1'] = tf.get_variable("W1", [self.n_obs, self.hidden_units], initializer=xavier_l1)
		with tf.variable_scope('layer_two', reuse=False):
			xavier_l2 = tf.truncated_normal_initializer(mean=0, stddev=1./np.sqrt(self.hidden_units), dtype=tf.float32)
			self.tf_model['W2'] = tf.get_variable("W2", [self.hidden_units , self.n_actions], initializer=xavier_l2)

		# tf placeholders
		self.tf_x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_obs], name="tf_x")
		self.tf_y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions], name="tf_y")
		self.tf_epr = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="tf_epr")

		self.tf_aprob = tf_policy_forward(self.tf_x , self.tf_model)

		self.sess = tf.InteractiveSession()
		self.saver = tf.train.Saver()

		if args.test_pg:
			#you can load your model here
			print('loading trained model')
			save_dir = 'pg_models'
			self.saver.restore(self.sess, 'pg_models/pong.ckpt-8500')
			# self.saver.restore(self.sess, tf.train.latest_checkpoint(save_dir))

		##################
		# YOUR CODE HERE #
		##################

	def init_game_setting(self):

		"""
		Testing function will call this function at the begining of new game
		Put anything you want to initialize if necessary
		"""

		##################
		# YOUR CODE HERE #
		##################
		pass

	def train(self):

		#Implement your training algorithm here

		##################
		# YOUR CODE HERE #
		##################
		gamma = 0.99              # discount factor for reward
		decay = 0.99              # decay rate for RMSProp gradients
		save_path='pg_models/pong.ckpt'

		observation = self.env.reset()
		xs, rs, ys = [], [], []
		running_reward = None
		reward_sum = 0
		reward_history = []
		episode_number = 0

		# tf reward processing (need tf_discounted_epr for policy gradient wizardry)
		tf_discounted_epr = tf_discount_rewards(self.tf_epr , gamma)
		tf_mean, tf_variance= tf.nn.moments(tf_discounted_epr, [0], shift=None, name="reward_moments")

		# Normalize
		tf_discounted_epr -= tf_mean
		tf_discounted_epr /= tf.sqrt(tf_variance + 1e-6)

		# tf optimizer op
		loss = tf.nn.l2_loss(self.tf_y - self.tf_aprob)
		optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=decay)
		tf_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=tf_discounted_epr)
		train_op = optimizer.apply_gradients(tf_grads)

		# tf graph initialization
		self.sess.run(tf.global_variables_initializer())

		# try load saved model
		try:
			load_path = tf.train.latest_checkpoint('pg_models')
			self.saver.restore(self.sess, load_path)
			with open('pg_models/reward_history.pkl', 'rb') as history:
				reward_history = pickle.load(history)
		except:
			print ("no saved model to load. starting new session")
		else:
			print ("loaded model: {}".format(load_path))
			episode_number = int(load_path.split('-')[-1])

		# training loop
		while True:
		    # if True: env.render()

			# preprocess the observation, set input to network to be difference image
			cur_x = prepro(observation)
			x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.n_obs)
			self.prev_x = cur_x

			# stochastically sample a policy from the network
			feed = {self.tf_x: np.reshape(x, (1,-1))}
			aprob = self.sess.run(self.tf_aprob, feed)
			aprob = aprob[0,:]
			action = np.random.choice(self.n_actions, p=aprob)
			label = np.zeros_like(aprob)
			label[action] = 1

			# step the environment and get new measurements
			observation, reward, done, info = self.env.step(action + 1)
			reward_sum += reward

			# record game history
			xs.append(x) ; ys.append(label) ; rs.append(reward)

			if done:
				# update running reward
				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				reward_history.append(reward_sum)

				# parameter update
				feed = {self.tf_x: np.vstack(xs), self.tf_y: np.vstack(ys), self.tf_epr: np.vstack(rs)}
				_ = self.sess.run(train_op, feed)

				# print progress console
				if episode_number % 10 == 0:
					print ('ep {}: reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward))
				else:
					print ('\tep {}: reward: {}'.format(episode_number, reward_sum))

				# bookkeeping
				xs, rs, ys = [], [], [] # reset game history
				episode_number += 1 # the Next Episode
				observation = self.env.reset() # reset env
				reward_sum = 0
				if episode_number % 100 == 0:
					self.saver.save(self.sess, save_path, global_step=episode_number)
					print ("SAVED MODEL #{}".format(episode_number))
					with open('pg_models/reward_history.pkl', 'wb') as history:
						pickle.dump(reward_history, history)



	def make_action(self, observation, test=True):
		"""
		Return predicted action of your agent

		Input:
			observation: np.array
				current RGB screen of game, shape: (210, 160, 3)

		Return:
			action: int
				the predicted action from trained model
		"""
		##################
		# YOUR CODE HERE #
		##################

		#if True: env.render()

		# preprocess the observation, set input to network to be difference image
		cur_x = prepro(observation)
		x = cur_x - self.prev_x if self.prev_x is not None else np.zeros(self.n_obs)
		self.prev_x = cur_x

		# stochastically sample a policy from the network
		feed = {self.tf_x: np.reshape(x, (1,-1))}
		aprob = self.sess.run(self.tf_aprob,feed) ; aprob = aprob[0,:]
		action = np.random.choice(self.n_actions, p=aprob)

		return action + 1
