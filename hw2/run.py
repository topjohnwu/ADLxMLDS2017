#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence

# params
batch_size = 64
feat_dim = 4096
hidden_dim = 256
frame_step = 80
caption_step = 40
learning_rate = 0.001
num_epoch = 6
word_count_threshold = 3

use_attention = False

# User inputs
mode = sys.argv[1]
model_name = sys.argv[2]
model_dir = os.path.join('models', model_name)
data_dir = sys.argv[3]

os.makedirs(model_dir, exist_ok=True)

# Global data
word_counts = {}
w2i = {}
i2w = {}
# b_init = np.array
train_labs = []
test_labs = []
train_feat = {}
test_feat = {}

def process_str(str):
	return str.lower().replace('.', '').replace(',', '').replace('!', '').replace('?', '').replace('"', '').replace('\'', '').replace('\\', '').replace('/', '').strip().split()

def load_data():
	with open('{}/training_label.json'.format(data_dir), 'r') as labels:
		labs = json.load(labels)
		for vid in labs:
			id = vid['id']
			captions = [ process_str(c) for c in vid['caption'] ]
			train_feat[id] = np.load('{}/training_data/feat/{}.npy'.format(data_dir, id))
			for sentence in captions:
				train_labs.append((id, sentence))
				for word in sentence:
					word_counts[word] = word_counts.get(word, 0) + 1

	with open('{}/testing_label.json'.format(data_dir), 'r') as labels:
		labs = json.load(labels)
		for vid in labs:
			id = vid['id']
			captions = [ process_str(c) for c in vid['caption'] ]
			test_feat[id] = np.load('{}/testing_data/feat/{}.npy'.format(data_dir, id))
			for sentence in captions:
				test_labs.append((id, sentence))
				for word in sentence:
					word_counts[word] = word_counts.get(word, 0) + 1

	word_counts['<pad>'] = len(train_labs)
	word_counts['<bos>'] = len(train_labs)
	word_counts['<eos>'] = len(train_labs)
	word_counts['<unk>'] = len(train_labs)

def build_model():
	with tf.variable_scope('embed'):
		emb_W = tf.get_variable('W', [vocab_size, hidden_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))

	with tf.variable_scope('encode'):
		enc_W = tf.get_variable('W', [feat_dim, hidden_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))
		enc_b = tf.get_variable('b', [hidden_dim], initializer=tf.constant_initializer(0.0))

	with tf.variable_scope('decode'):
		dec_W = tf.get_variable('W', [hidden_dim, vocab_size], initializer=tf.random_uniform_initializer(-0.1, 0.1))
		dec_b = tf.Variable(b_init.astype(np.float32), name='b')

	with tf.variable_scope('attention'):
		att_W = tf.get_variable('W', [hidden_dim, hidden_dim], initializer=tf.random_uniform_initializer(-0.1, 0.1))

	lstm1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)
	lstm2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_dim, state_is_tuple=True)

	return emb_W, enc_W, enc_b, dec_W, dec_b, lstm1, lstm2, att_W

def build_train_model():
	emb_W, enc_W, enc_b, dec_W, dec_b, lstm1, lstm2, att_W = build_model()

	video = tf.placeholder(tf.float32, [batch_size, frame_step, feat_dim], name='video')
	caption = tf.placeholder(tf.int32, [batch_size, caption_step + 1], name='caption')
	caption_mask = tf.placeholder(tf.float32, [batch_size, caption_step + 1], name='caption_mask')

	video_flat = tf.reshape(video, [-1, feat_dim])
	image_emb = tf.nn.xw_plus_b(video_flat, enc_W, enc_b)
	image_emb = tf.reshape(image_emb, [batch_size, -1, hidden_dim])

	state1 = list(map(lambda x : tf.zeros([batch_size, x]), lstm1.state_size))
	state2 = list(map(lambda x : tf.zeros([batch_size, x]), lstm2.state_size))
	padding = tf.zeros([batch_size, hidden_dim])

	loss = 0.0
	enc_outputs = []

	with tf.variable_scope(tf.get_variable_scope()) as scope:
		for i in range(0, frame_step):
			if i > 0:
				scope.reuse_variables()

			with tf.variable_scope("LSTM1"):
				output1, state1 = lstm1(image_emb[:,i,:], state1)

			enc_outputs.append(output1)

			with tf.variable_scope("LSTM2"):
				output2, state2 = lstm2(tf.concat([padding, output1], axis=1), state2)

		enc_outputs = tf.convert_to_tensor(enc_outputs)

		for i in range(0, caption_step):
			current_embed = tf.nn.embedding_lookup(emb_W, caption[:, i])

			scope.reuse_variables()

			with tf.variable_scope("LSTM1"):
				output1, state1 = lstm1(padding, state1)

			if use_attention:
				weighted_output1 = []
				for j in range(batch_size):
					ctx = enc_outputs[:, j, :] @ att_W @ tf.expand_dims(output1[j, :], axis=1)
					ctx = tf.nn.softmax(tf.squeeze(ctx))
					ctx = tf.reshape(ctx, [1, -1])
					weighted_output1.append(tf.squeeze(ctx @ enc_outputs[:, j, :]))
			else:
				weighted_output1 = output1

			with tf.variable_scope("LSTM2"):
				output2, state2 = lstm2(tf.concat([current_embed, weighted_output1], axis=1), state2)

			labels = tf.expand_dims(caption[:, i + 1], axis=1)
			indices = tf.expand_dims(tf.range(0, batch_size), axis=1)
			concated = tf.concat([indices, labels], axis=1)
			onehot_labels = tf.sparse_to_dense(concated, [batch_size, vocab_size], 1.0, 0.0)

			logit_words = tf.nn.xw_plus_b(output2, dec_W, dec_b)
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
			cross_entropy = cross_entropy * caption_mask[:, i]

			current_loss = tf.reduce_sum(cross_entropy) / batch_size
			loss += current_loss

	opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	return video, caption, caption_mask, loss, opt

def build_test_model():
	emb_W, enc_W, enc_b, dec_W, dec_b, lstm1, lstm2, att_W = build_model()

	video = tf.placeholder(tf.float32, [frame_step, feat_dim], name='video')

	video_flat = tf.reshape(video, [-1, feat_dim])
	image_emb = tf.nn.xw_plus_b(video_flat, enc_W, enc_b)
	image_emb = tf.reshape(image_emb, [1, -1, hidden_dim])

	state1 = list(map(lambda x : tf.zeros([1, x]), lstm1.state_size))
	state2 = list(map(lambda x : tf.zeros([1, x]), lstm2.state_size))
	padding = tf.zeros([1, hidden_dim])

	generated_word_index = []
	enc_outputs = []

	with tf.variable_scope(tf.get_variable_scope()) as scope:
		for i in range(0, frame_step):
			if i > 0:
				scope.reuse_variables()

			with tf.variable_scope("LSTM1"):
				output1, state1 = lstm1(image_emb[:, i, :], state1)

			enc_outputs.append(output1)

			with tf.variable_scope("LSTM2"):
				output2, state2 = lstm2(tf.concat([padding, output1], axis=1), state2)

		enc_outputs = tf.convert_to_tensor(enc_outputs)

		for i in range(0, caption_step):
			scope.reuse_variables()

			if i == 0:
				current_embed = tf.nn.embedding_lookup(emb_W, tf.ones([1], dtype=tf.int64))

			with tf.variable_scope("LSTM1"):
				output1, state1 = lstm1(padding, state1)

			if use_attention:
				weighted_output1 = []
				ctx = enc_outputs[:, 0, :] @ att_W @ tf.expand_dims(output1[0, :], axis=1)
				ctx = tf.nn.softmax(tf.squeeze(ctx))
				ctx = tf.reshape(ctx, [1, -1])
				weighted_output1.append(tf.squeeze(ctx @ enc_outputs[:, 0, :]))
			else:
				weighted_output1 = output1

			with tf.variable_scope("LSTM2"):
				output2, state2 = lstm2(tf.concat([current_embed, weighted_output1], axis=1), state2)

			logit_words = tf.nn.xw_plus_b( output2, dec_W, dec_b)
			word_idx = tf.argmax(logit_words, 1)[0]
			generated_word_index.append(word_idx)

			current_embed = tf.nn.embedding_lookup(emb_W, word_idx)
			current_embed = tf.expand_dims(current_embed, 0)

		return video, generated_word_index

def cap_to_idx(caption):
	return [1] + [ w2i.get(w, 3) for w in caption[:caption_step - 2] ] + [2]

def train():
	global i2w, w2i, b_init, vocab_size

	# Filter vocab
	vocab = [ v for v in word_counts if word_counts[v] >= word_count_threshold ]

	i2w[0] = '<pad>'
	i2w[1] = '<bos>'
	i2w[2] = '<eos>'
	i2w[3] = '<unk>'
	w2i['<pad>'] = 0
	w2i['<bos>'] = 1
	w2i['<eos>'] = 2
	w2i['<unk>'] = 3
	for i, w in enumerate(vocab):
		w2i[w] = i + 4
		i2w[i + 4] = w

	b_init = np.array([1.0 * word_counts[ i2w[i] ] for i in i2w])
	b_init /= np.sum(b_init) # normalize to frequencies
	b_init = np.log(b_init)
	b_init -= np.max(b_init) # shift to nice numeric range

	# Dump saved
	np.save('{}/i2w.npy'.format(model_dir), i2w)
	np.save('{}/w2i.npy'.format(model_dir), w2i)
	np.save('{}/b_init.npy'.format(model_dir), b_init)

	vocab_size = len(i2w)

	tf_video, tf_caption, tf_caption_mask, tf_loss, tf_opt = build_train_model()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()

	tf.global_variables_initializer().run()
	batch_feats = np.empty([batch_size, frame_step, feat_dim])
	for epoch in range(num_epoch):
		np.random.shuffle(train_labs)

		for idx in range(0, len(train_labs) - batch_size, batch_size):
			batch_captions = train_labs[idx:idx + batch_size]
			for n, sentence in enumerate(batch_captions):
				batch_feats[n] = train_feat[sentence[0]]

			batch_captions_idx = sequence.pad_sequences([ cap_to_idx(cap[1]) for cap in batch_captions ], padding='post', maxlen=caption_step + 1)
			batch_caption_masks = np.zeros(batch_captions_idx.shape)

			nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, batch_captions_idx )) )
			for ind, row in enumerate(batch_caption_masks):
				row[:nonzeros[ind]] = 1

			_, loss_val = sess.run([tf_opt, tf_loss],feed_dict={ tf_video: batch_feats, tf_caption: batch_captions_idx, tf_caption_mask: batch_caption_masks})
			print("epoch=[{}] idx=[{}] loss=[{}]".format(epoch, idx, loss_val))

		saver.save(sess, '{}/model'.format(model_dir))

def test(file_list = None):
	global i2w, w2i, b_init, vocab_size

	# Load saved
	i2w = np.load('{}/i2w.npy'.format(model_dir)).tolist()
	w2i = np.load('{}/w2i.npy'.format(model_dir))
	b_init = np.load('{}/b_init.npy'.format(model_dir))

	vocab_size = len(i2w)

	tf_video, tf_caption = build_test_model()
	sess = tf.InteractiveSession()
	saver = tf.train.Saver()

	saver.restore(sess, '{}/model'.format(model_dir))

	if file_list == None:
		file_list = test_feat

	outfile = open(sys.argv[4], 'w')

	for id in file_list:
		generated_word_index = sess.run(tf_caption, feed_dict={ tf_video: test_feat[id] })
		generated_words = [ i2w[i] for i in generated_word_index ]
		idx = generated_words.index('<eos>')
		sentence = ' '.join([ w for w in generated_words[:idx] if w != '<unk>' ])
		str = '{},{}'.format(id, sentence)
		print(str)
		outfile.write(str)
		outfile.write('\n')

	outfile.close()

load_data()

if mode == 'train':
	train()
elif mode == 'test':
	test()
else:
	test(['klteYv1Uv9A_27_33.avi', '5YJaS2Eswg0_22_26.avi', 'UbmZAe5u5FI_132_141.avi', 'JntMAcTlOF0_50_70.avi', 'tJHUH9tpqPg_113_118.avi'])
