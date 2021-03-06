#!/usr/bin/env python3

import sys
import numpy as np
import tensorflow as tf
import random

# Consts
train_frames = 1124823
test_frames = 180406
fbank_dim = 69
mfcc_dim = 39
phone_num = 48

# Params
num_steps = 20
state_size = 100
learning_rate = 0.001
num_epochs = 20
rnn_layers = 3

# CNN params
cnn_kernel_size = 3
cnn_filter_num = [32, 64]
fc_out_num = [1024, phone_num]
prob = 0.9


# User inputs
train = sys.argv[1] == 'train'
CNN = sys.argv[2] == 'cnn'
batch_size = int(sys.argv[3])
data_dir = sys.argv[4]
model_name = sys.argv[5]

metadata = []
slots = []
label_dic = {}
o2o_map = {}
num_39_map = {}

# placeholders
x = tf.placeholder(tf.float32, [batch_size, num_steps, fbank_dim], name='input')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

if CNN:
	# Construct CNN
	conv_x = tf.pad(tf.expand_dims(x, axis=3), [[0, 0], [1, 1], [0, 0], [0, 0]], 'SYMMETRIC')
	cnn_list = []

	for i in range(1, num_steps + 1):
		sliced = conv_x[:, i - 1:i + 2, :]

		# CNN 1
		w_conv1 = tf.Variable(tf.truncated_normal([cnn_kernel_size, cnn_kernel_size, 1, cnn_filter_num[0]], stddev=0.1))
		b_conv1 = tf.Variable(tf.constant(0.1, shape=[cnn_filter_num[0]]))
		conv1 = tf.nn.relu(tf.nn.conv2d(sliced, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
		pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

		# CNN 2
		# w_conv2 = tf.truncated_normal([cnn_kernel_size, cnn_kernel_size, cnn_filter_num[0], cnn_filter_num[1]], stddev=0.1)
		# b_conv2 = tf.constant(0.1, shape=[cnn_filter_num[1]])
		# conv2 = tf.nn.relu(tf.nn.conv2d(pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
		# pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

		flat_dim = int(np.prod(pool1.shape[1:]))
		flatten = tf.reshape(pool1, [-1, flat_dim])

		# FC 1
		w_fc1 = tf.Variable(tf.truncated_normal([flat_dim, fc_out_num[0]], stddev=0.1))
		b_fc1 = tf.Variable(tf.constant(0.1, shape=[fc_out_num[0]]))
		fc1 = tf.nn.relu(tf.matmul(flatten, w_fc1) + b_fc1)
		fc1_drop = tf.nn.dropout(fc1, keep_prob)

		# FC 2
		w_fc2 = tf.Variable(tf.truncated_normal([fc_out_num[0], fc_out_num[1]], stddev=0.1))
		b_fc2 = tf.Variable(tf.constant(0.1, shape=[fc_out_num[1]]))
		fc2 = tf.nn.relu(tf.matmul(fc1_drop, w_fc2) + b_fc2)
		fc2_drop = tf.nn.dropout(fc2, keep_prob)

		cnn_list.append(fc2_drop)

	rnn_input = tf.stack(cnn_list, axis=1)
else:
	rnn_input = x

# cell
cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True) for i in range(rnn_layers)], state_is_tuple=True)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_input, dtype=tf.float32)

# logits and predictions
with tf.variable_scope('softmax'):
	W = tf.get_variable('W', [state_size, phone_num], dtype=tf.float32)
	b = tf.get_variable('b', [phone_num], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
logits = tf.reshape(tf.matmul(tf.reshape(rnn_outputs, [-1, state_size]), W) + b, [batch_size, num_steps, phone_num])
predictions = tf.nn.softmax(logits)
last_label = tf.argmax(predictions, axis=2)[:,-1]

#losses and train_step
one_hot_label = tf.one_hot(y, phone_num, on_value=1.0, off_value=0.0)
losses = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_label, logits=logits)
total_loss = tf.reduce_mean(losses)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, axis=2), tf.argmax(one_hot_label, axis=2)), tf.float32))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

saver = tf.train.Saver()

def gen_train_slot():
	random.shuffle(slots)

	for slot in slots:
		slot_x = [data_fbank[n[1]] for n in metadata[slot[0]][slot[1]:slot[1] + num_steps]]
		slot_y = [o2o_map[n[2]][0] for n in metadata[slot[0]][slot[1]:slot[1] + num_steps]]
		yield((slot_x, slot_y))

def gen_train_batch():
	idx = 0
	data_x = np.empty([batch_size, num_steps, fbank_dim], dtype=np.float32)
	data_y = np.empty([batch_size, num_steps], dtype=np.int32)
	slot_generator = gen_train_slot()
	for _ in range(0, len(slots) // batch_size):
		for i in range(0, batch_size):
			data_x[i], data_y[i] = next(slot_generator)
		yield((data_x, data_y))

def gen_train_epochs():
	for i in range(num_epochs):
		yield gen_train_batch()

def gen_test_slot(sentence):
	for frame in range(0, len(sentence)):
		if frame < num_steps:
			slot_x = [data_fbank[sentence[0][1]] for _ in range(num_steps - frame - 1)]
			slot_x.extend([data_fbank[n[1]] for n in sentence[0:frame + 1]])
		else:
			slot_x = [data_fbank[n[1]] for n in sentence[frame - num_steps + 1:frame + 1]]
		yield(slot_x)
	while True:
		# Start providing unlimited zeros
		slot_x = [[0.0 for _ in range(fbank_dim)] for __ in range(num_steps)]
		yield(slot_x)

def gen_test_batch(sentence):
	data_x = np.empty([batch_size, num_steps, fbank_dim], dtype=np.float32)
	slot_generator = gen_test_slot(sentence)
	# If the frame number is larger than a batch
	for _ in range(len(sentence) // batch_size):
		for i in range(batch_size):
			data_x[i] = next(slot_generator)
		yield(data_x)
	# Last batch (with padding)
	if len(sentence) % batch_size != 0:
		# Create a padded batch
		for i in range(batch_size):
			data_x[i] = next(slot_generator)
		yield(data_x)

def get_output(num_list, consec_thold = 3):
	trimmed = []
	state = 0
	prev = ''
	for c in [num_39_map[num] for num in num_list]:
		if c == prev:
			state += 1
		else:
			prev = c
			state = 0
		if state >= consec_thold - 1:
			if len(trimmed) == 0 or trimmed[-1] != c:
				trimmed.append(c)
	# Trim start and end sil
	if trimmed[0] == 'L':
		trimmed = trimmed[1:]
	if trimmed[-1] == 'L':
		trimmed = trimmed[:-1]
	return ''.join(trimmed)

# Read mapping
with open('{}/48phone_char.map'.format(data_dir), 'r') as mappings:
	for line in mappings:
		arr = line.strip('\n\r').split('\t')
		o2o_map[arr[0]] = (arr[1], arr[2])

with open('{}/phones/48_39.map'.format(data_dir), 'r') as mappings:
	for line in mappings:
		arr = line.strip('\n\r').split('\t')
		num_39_map[int(o2o_map[arr[0]][0])] = o2o_map[arr[1]][1]

if train:
	# Read label
	with open('{}/label/train.lab'.format(data_dir), 'r') as labels:
		for line in labels:
			arr = line.strip('\n\r').split(',')
			label_dic[arr[0]] = arr[1]

	# Read ARK
	data_fbank = np.empty([train_frames, fbank_dim], dtype=np.float32)
	with open('{}/fbank/train.ark'.format(data_dir), 'r') as fbank_train:
		name = ''
		sentence = []
		for idx, line in enumerate(fbank_train):
			arr = line.split(' ')
			s = arr[0][:arr[0].rfind('_')]
			if name != s:
				if sentence != []:
					metadata.append(sentence)
				sentence = []
				name = s
			sentence.append((arr[0], idx, label_dic[arr[0]]))
			data_fbank[idx] = arr[1:]
		metadata.append(sentence)

	# Gen slot list
	for idx, sentence in enumerate(metadata):
		slots.extend([[idx, x] for x in range(len(sentence) - num_steps + 1)])

	# Train
	with tf.Session() as sess:
		# saver.restore(sess, 'models/{}'.format(model_name))
		sess.run(tf.global_variables_initializer())
		for idx, epoch in enumerate(gen_train_epochs()):
			print('')
			for step, (X, Y) in enumerate(epoch):
				_ , acc = sess.run([train_step, accuracy], feed_dict={x:X, y:Y, keep_prob:prob})
				print('epoch=[{}] step=[{}] acc=[{}]'.format(idx, step, acc))
			# Save each epoch
			save_path = saver.save(sess, 'models/{}'.format(model_name))
else:
	# Read ARK
	data_fbank = np.empty([test_frames, fbank_dim], dtype=np.float32)
	with open('{}/fbank/test.ark'.format(data_dir), 'r') as fbank_test:
		name = ''
		sentence = []
		for idx, line in enumerate(fbank_test):
			arr = line.split(' ')
			s = arr[0][:arr[0].rfind('_')]
			if name != s:
				if sentence != []:
					metadata.append(sentence)
				sentence = []
				name = s
			sentence.append((arr[0], idx))
			data_fbank[idx] = arr[1:]
		metadata.append(sentence)

	csv = open(sys.argv[6], 'w')
	csv.write('id,phone_sequence\n')

	# Test
	with tf.Session() as sess:
		saver.restore(sess, 'models/{}'.format(model_name))
		for sentence in metadata:
			sen_len = len(sentence)
			num_list = []
			for X in gen_test_batch(sentence):
				pred = sess.run(last_label, feed_dict={x:X, keep_prob:prob})
				new_list = pred[:sen_len]
				num_list.extend(new_list)
				sen_len -= len(new_list)
			s = ','.join((sentence[0][0][:sentence[0][0].rfind('_')], get_output(num_list)))
			print(s)
			csv.write(s)
			csv.write('\n')
	csv.close()
