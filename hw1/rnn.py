#!/usr/bin/env python3

import sys
import numpy as np
import tensorflow as tf

# Consts
train_frames = 1124823
test_frames = 180406
fbank_dim = 69
mfcc_dim = 39
phone_num = 48

# Params
num_steps = 20
batch_size = 500
state_size = 100
learning_rate = 0.001
num_epochs = 20
layers = 3

train = sys.argv[1] == 'train'
data_dir = sys.argv[2]
model_name = sys.argv[3]

metadata = []
label_dic = {}
o2o_map = {}
num_39_map = {}

# placeholders
x = tf.placeholder(tf.float32, [batch_size, num_steps, fbank_dim], name='input')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels')
init_state = tf.zeros([batch_size, state_size], dtype=tf.float32)

# cell
cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True) for i in range(layers)], state_is_tuple=True)
rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

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
	for sentence in metadata:
		for frame in range(0, len(sentence) - num_steps + 1):
			slot_x = [data_fbank[n[1]] for n in sentence[frame:frame + num_steps]]
			slot_y = [o2o_map[n[2]][0] for n in sentence[frame:frame + num_steps]]
			yield((slot_x, slot_y))

def gen_train_batch():
	idx = 0
	data_x = np.empty([batch_size, num_steps, fbank_dim], dtype=np.float32)
	data_y = np.empty([batch_size, num_steps], dtype=np.int32)
	# Calc total slots
	slots = 0
	for sentence in metadata:
		slots += (len(sentence) - num_steps + 1)
	print('Total availible slots=[{}]'.format(slots))
	slot_generator = gen_train_slot()
	for _ in range(0, slots // batch_size):
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
with open('{}/phones/48phone_char.map'.format(data_dir), 'r') as mappings:
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

	# Train
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for idx, epoch in enumerate(gen_train_epochs()):
			print('')
			for step, (X, Y) in enumerate(epoch):
				_ , acc = sess.run([train_step, accuracy], feed_dict={x:X, y:Y})
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

	csv = open(sys.argv[4], 'w')
	csv.write('id,phone_sequence\n')

	# Test
	with tf.Session() as sess:
		saver.restore(sess, 'models/{}'.format(model_name))
		for sentence in metadata:
			sen_len = len(sentence)
			# print('{} [{}]'.format(sentence[0][0][:sentence[0][0].rfind('_')], sen_len))
			num_list = []
			for X in gen_test_batch(sentence):
				pred = sess.run(last_label, feed_dict={x:X})
				num_list.extend(pred[:sen_len])
				sen_len -= len(num_list)
			csv.write(','.join((sentence[0][0][:sentence[0][0].rfind('_')], get_output(num_list))))
			csv.write('\n')
	csv.close()
