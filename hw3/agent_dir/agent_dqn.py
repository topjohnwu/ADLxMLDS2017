from agent_dir.agent import Agent
import tensorflow as tf
import numpy as np
import pickle
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def conv2d_layer(input_tensor, filter_shape, strides, name, collections, padding='SAME', activation=tf.nn.relu):
	with tf.variable_scope(name):
		filters = tf.get_variable('filters', filter_shape, dtype=tf.float32, collections=collections)
		bias = tf.get_variable('b', [filter_shape[3]], dtype=tf.float32, collections=collections)
		conv = tf.nn.conv2d(input_tensor, filters, strides, padding)
		output = activation(tf.nn.bias_add(conv, bias))
		return output

def dense_layer(input_tensor, input_dim, output_dim, name, collections, activation=tf.nn.relu):
	with tf.variable_scope(name):
		W = tf.get_variable('W', [input_dim, output_dim], dtype=tf.float32, collections=collections)
		b = tf.get_variable('b', [output_dim], dtype=tf.float32, collections=collections)
		output = activation(tf.matmul(input_tensor, W))
		return output

def lrelu(x, alpha=0.2):
	return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

class Agent_DQN(Agent):
	def __init__(self, env, args):
		"""
		Initialize every things you need here.
		For example: building your model
		"""

		super(Agent_DQN,self).__init__(env)

		self.n_actions = env.action_space.n
		self.learning_rate = 0.0001
		self.gamma = 0.99
		self.batch_size = 32
		self.memory_size = 10000
		self.memory_count = 0
		self.learn_step_count = 0
		self.update_target_freq = 250
		if args.test_dqn:
			self.epsilon = 0
			self.epsilon_decay = 0
			self.epsilon_min = 0
		else:
			self.epsilon = 1
			self.epsilon_decay = 1e-6
			self.epsilon_min = 0.05

		self.states = np.zeros((self.memory_size, 84, 84, 4))
		self.actions = np.zeros(self.memory_size)
		self.rewards = np.zeros(self.memory_size)
		self.next_states = np.zeros((self.memory_size, 84, 84, 4))
		self.done = np.zeros(self.memory_size)

		####################### Q Network ##########################
		self.tf_states = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states')

		with tf.variable_scope('Q_network'):
			collection_name = ['Q_network', tf.GraphKeys.GLOBAL_VARIABLES]
			conv1 = conv2d_layer(self.tf_states, [8, 8, 4, 32], [1, 4, 4, 1], 'conv1', collection_name)
			conv2 = conv2d_layer(conv1, [4, 4, 32, 64], [1, 2, 2, 1], 'conv2', collection_name)
			conv3 = conv2d_layer(conv2, [3, 3, 64, 64], [1, 1, 1, 1], 'conv3', collection_name)
			flatten_dim = np.prod(conv3.get_shape().as_list()[1:])
			flatten = tf.reshape(conv3, [-1, flatten_dim])
			dense = dense_layer(flatten, flatten_dim, 512, 'dense', collection_name, lrelu)
			self.Q_value = dense_layer(dense, 512, self.n_actions, 'output', collection_name, tf.identity)

		#################### Target Q Network ######################
		self.tf_states_t = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states_target')

		with tf.variable_scope('target_network'):
			collection_name = ['target_network', tf.GraphKeys.GLOBAL_VARIABLES]
			conv1 = conv2d_layer(self.tf_states_t, [8, 8, 4, 32], [1, 4, 4, 1], 'conv1', collection_name)
			conv2 = conv2d_layer(conv1, [4, 4, 32, 64], [1, 2, 2, 1], 'conv2', collection_name)
			conv3 = conv2d_layer(conv2, [3, 3, 64, 64], [1, 1, 1, 1], 'conv3', collection_name)
			flatten_dim = np.prod(conv3.get_shape().as_list()[1:])
			flatten = tf.reshape(conv3, [-1, flatten_dim])
			dense = dense_layer(flatten, flatten_dim, 512, 'dense', collection_name, lrelu)
			self.Q_target = dense_layer(dense, 512, self.n_actions, 'output', collection_name, tf.identity)

		################### Loss and Optimizer #####################
		self.tf_actions = tf.placeholder(tf.int32, [None], name='actions')
		self.tf_y = tf.placeholder(tf.float32, [None], name='y')

		Q_action = tf.reduce_sum(tf.multiply(self.Q_value, tf.one_hot(self.tf_actions, self.n_actions)), axis=-1)
		self.loss = tf.reduce_mean(tf.squared_difference(self.tf_y, Q_action))
		self.train_step = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

		self.sess  = tf.Session()
		self.saver = tf.train.Saver()

		q_params = tf.get_collection('Q_network')
		t_params = tf.get_collection('target_network')
		self.update_target_op = [tf.assign(t, q) for t, q in zip(t_params, q_params)]

		self.loss_record = []

		if args.test_dqn:
			#you can load your model here
			print('loading trained model')
			# self.saver.restore(self.sess, tf.train.latest_checkpoint('dqn_models'))
			self.saver.restore(self.sess, 'dqn_models/breakout.ckpt-4300000')
			with open('dqn_models/records.pkl', 'rb') as records:
				_, self.learn_step_count, self.epsilon, _ = pickle.load(records)

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
		"""
		Implement your training algorithm here
		"""
		##################
		# YOUR CODE HERE #
		##################

		start_steps = 10000
		frame_count = 0
		learn_count = 0
		episode_count = 0
		episode_reward = 0
		learn_freq = 4
		step_count = 0
		reward_history = []

		self.sess.run(tf.global_variables_initializer())
		state = self.env.reset()

		# try load saved model
		try:
			load_path = tf.train.latest_checkpoint('dqn_models')
			self.saver.restore(self.sess, load_path)
		except:
			print ("no saved model to load. starting new session")
		else:
			print ("loaded model: {}".format(load_path))
			with open('dqn_models/records.pkl', 'rb') as records:
				step_count, self.learn_step_count, self.epsilon, reward_history = pickle.load(records)
			episode_count = len(reward_history)


		while True:
			action = self.make_action(state)
			next_state, reward, done, info = self.env.step(action)
			index = self.memory_count % self.memory_size
			self.states[index] = state
			self.actions[index] = action
			self.rewards[index] = reward
			self.next_states[index] = next_state
			self.done[index] = done
			self.memory_count += 1
			state = next_state

			if (step_count >= start_steps) and (step_count % learn_freq == 0):
				if self.learn_step_count % self.update_target_freq == 0:
					_ = self.sess.run(self.update_target_op)

				# Sample a batch from memory
				if self.memory_count > self.memory_size:
					indices = np.random.choice(self.memory_size, size=self.batch_size)
				else:
					indices = np.random.choice(self.memory_count, size=self.batch_size)
				state_batch = self.states[indices]
				action_batch = self.actions[indices]
				reward_batch = self.rewards[indices]
				next_state_batch = self.next_states[indices]
				done_batch = self.done[indices]

				# Calculate Target Q values
				Q_target = self.sess.run(self.Q_target, feed_dict={self.tf_states_t: next_state_batch})

				# Calculate y
				y_batch = [reward_batch[i] if done_batch[i] else reward_batch[i] + self.gamma * np.max(Q_target[i]) for i in range(self.batch_size)]

				# Update parameters
				_, loss = self.sess.run([self.train_step, self.loss], feed_dict={self.tf_states: state_batch,
																				 self.tf_actions: action_batch,
																				 self.tf_y: y_batch})
				# self.loss_record.append(loss)
				self.learn_step_count += 1
				# learn_count += 1
			
			step_count += 1
			frame_count += 1
			episode_reward += reward

			if done:
				episode_count += 1
				sys.stdout.write('\rEpisode # %-5d | steps: %-5d | episode reward: %f\n' % (episode_count, frame_count, episode_reward))
				sys.stdout.flush()
				reward_history.append(episode_reward)
				frame_count = 0
				episode_reward = 0
				state = self.env.reset()

			sys.stdout.write('\r#--- Step: %-7d  epsilon: %-6.4f  updated times: %-6d' % (step_count, self.epsilon, self.learn_step_count))

			if (step_count >= 10000) and (step_count % 100000 == 0):
				print('\nSAVED MODEL: #{}'.format(step_count))
				self.saver.save(self.sess, 'dqn_models/breakout.ckpt', global_step=step_count)
				with open('dqn_models/records.pkl', 'wb') as records:
					pickle.dump([step_count, self.learn_step_count, self.epsilon, reward_history], records)

				# if (step_count > 2000000) and (step_count % 500000 == 0):
				#     test_env = Environment('BreakoutNoFrameskip-v4', self.args, atari_wrapper=True, test=True)
				#     test_score = test(self, test_env, 100)
				#     print('################################################################                                   ')
				#     print('Step: %d  | testing score: %f' % (step_count, test_score))
				#     print('################################################################')
				#     if test_score > 40:
				#         wait = input("Baseline passed. Continue? [y/n] ")
				#         if wait == n: break


	def make_action(self, observation, test=True):
		"""
		Return predicted action of your agent

		Input:
			observation: np.array
				stack 4 last preprocessed frames, shape: (84, 84, 4)

		Return:
			action: int
				the predicted action from trained model
		"""
		##################
		# YOUR CODE HERE #
		##################
		if np.random.uniform() < self.epsilon:
			action = np.random.choice(self.n_actions)
		else:
			action_values = self.sess.run(self.Q_value, feed_dict={self.tf_states: np.expand_dims(observation, 0)})
			action = np.argmax(action_values)
		self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min
		return action

