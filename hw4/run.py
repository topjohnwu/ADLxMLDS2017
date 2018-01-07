#! /usr/bin/env python3
import tensorflow as tf
import numpy as np
import skimage.io
import skimage.transform
import os
import DCGAN
import pickle
import sys

# feature 0 ~ 11
hair_colors = ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 'pink', 'blue', 'black', 'brown', 'blonde']

# feature 12 ~ 22
eye_colors = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 'brown', 'red', 'blue']

num_epochs = 500
learning_rate = 0.0001
resume_model = False
image_dir = 'faces'
csv = 'tags_clean.csv'
txt = 'testing_text.txt'
pregen_feature = 'features.pkl'
loss_ckpt = 'losses.pkl'
dis_updates = 1
gen_updates = 2
model_path = 'models/dcgan'
save_path = 'samples'
sample_num = 1

"""
z_dim: noise dimension
t_dim: text  dimension
gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
caption_length: caption vector length
"""

params = dict(
    z_dim = 100,
    t_dim = 256,
    batch_size = 32,
    image_size = 64,
    gf_dim = 64,
    df_dim = 64,
    gfc_dim = 1024,
    dfc_dim = 1024,
    istrain = sys.argv[1] == '--train',
    caption_length = len(hair_colors) + len(eye_colors)
)

def gen_training_batch(params, images , embeddings):
    real_images = np.empty((params['batch_size'], params['image_size'], params['image_size'], 3))
    wrong_images = np.empty((params['batch_size'], params['image_size'], params['image_size'], 3))
    real_captions = np.empty((params['batch_size'], params['caption_length']))
    # wrong_captions = np.empty((params['batch_size'], params['caption_length']))

    for batch_num in range(len(images) // params['batch_size']):
        real_idx = np.arange(batch_num * params['batch_size'], (batch_num + 1) * params['batch_size'], dtype=int)
        real_images = images[real_idx]
        real_captions = embeddings[real_idx]

        rand_idx = np.random.randint(len(images), size=params['batch_size'])
        wrong_images = images[rand_idx]

        # rand_idx = np.random.randint(len(images), size=params['batch_size'])
        # wrong_captions = embeddings[rand_idx]

        z_noise = np.random.uniform(-1, 1, [params['batch_size'], params['z_dim']])

        # yield real_images, real_captions, wrong_images, wrong_captions, z_noise
        yield real_images, real_captions, wrong_images, z_noise

def read_images(image_dir , csv , params):
    images = []
    captions = []

    with open(csv, 'r') as f:
        for line in f:
            feature_vector = np.zeros(params['caption_length'])
            id, tags = line.rstrip().split(',')
            tags = [ tag.split(':')[0] for tag in tags.split('\t') ]
            for tag in tags:
                if 'hair' in tag:
                    try:
                        feature_vector[hair_colors.index(tag.split(' ')[0])] = 1
                    except:
                        pass
                elif 'eye' in tag:
                    try:
                        feature_vector[eye_colors.index(tag.split(' ')[0]) + 12] = 1
                    except:
                        pass
            # At least one feature exists
            if (np.sum(feature_vector) > 0):
                img = skimage.io.imread(os.path.join(image_dir, id + '.jpg'))
                img_resized = skimage.transform.resize(img, (params['image_size'], params['image_size']), mode='constant')
                images.append(img_resized)
                captions.append(feature_vector)

    # Pad to fit batch size
    padding = params['batch_size'] - (len(images) % params['batch_size'])
    for i in range(padding):
        images.append(images[i])
        captions.append(captions[i])

    return np.array(images), np.array(captions)

def gen_testing_batch(txt, params):
    with open(txt, 'r') as f:
        for line in f:
            feature_vector = np.zeros(params['caption_length'])
            id, tags = line.rstrip().split(',')
            tags = tags.split(' ')
            feature_vector[hair_colors.index(tags[0])] = 1
            feature_vector[eye_colors.index(tags[2])] = 1
            z_noise = np.random.uniform(-1, 1, [params['batch_size'], params['z_dim']])
            yield id, np.tile(feature_vector, [params['batch_size'], 1]), z_noise

gan = DCGAN.GAN(params)
input_tensors, variables, outputs, loss = gan.build_model()
d_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(loss['d_loss'], var_list=variables['d_vars'])
g_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5).minimize(loss['g_loss'], var_list=variables['g_vars'])

sess = tf.InteractiveSession()
saver = tf.train.Saver()

if params['istrain']:
    try:
        with open(pregen_feature, 'rb') as features:
            images, captions = pickle.load(features)
            print('Loaded pre-generated features')
    except:
        print('Generating features')
        images, captions = read_images(image_dir , csv , params)
        with open(pregen_feature, 'wb') as features:
            pickle.dump([images, captions], features)

    # load saved model
    epoch_num = 0
    d_loss_mean = []
    g_loss_mean = []
    try:
        load_path = tf.train.latest_checkpoint(os.path.dirname(model_path))
        saver.restore(sess, load_path)
        with open(loss_ckpt, 'rb') as losses:
            d_loss_mean, g_loss_mean = pickle.load(losses)
    except:
        print ("No saved model to load, starting new session")
        sess.run(tf.global_variables_initializer())
    else:
        print ("Loaded model: {}".format(load_path))
        epoch_num = int(load_path.split('-')[-1])

    while True:
        batch_num = 0
        d_loss_sum = 0
        g_loss_sum = 0
        for real_images, real_captions, wrong_images, z_noise in gen_training_batch(params, images, captions):
            # dis update
            for _ in range(dis_updates):
                _, d_loss = sess.run([d_opt, loss['d_loss']],
                        feed_dict = {
                            input_tensors['t_real_image'] : real_images,
                            input_tensors['t_wrong_image'] : wrong_images,
                            input_tensors['t_real_caption'] : real_captions,
                            input_tensors['t_z'] : z_noise,
                        })

            # gen update
            for _ in range(gen_updates):
                _, g_loss = sess.run([g_opt, loss['g_loss']],
                        feed_dict = {
                            input_tensors['t_real_image'] : real_images,
                            input_tensors['t_wrong_image'] : wrong_images,
                            input_tensors['t_real_caption'] : real_captions,
                            input_tensors['t_z'] : z_noise,
                        })

            d_loss_sum += d_loss
            g_loss_sum += g_loss

            batch_num += 1
            sys.stdout.write('\rd_loss = {:5f} g_loss = {:5f} batch_num = {:3d} epochs = {}   '.format(d_loss_sum / batch_num, g_loss_sum / batch_num, batch_num, epoch_num))
            sys.stdout.flush()

        epoch_num += 1
        d_loss_mean.append(d_loss_sum / batch_num)
        g_loss_mean.append(g_loss_sum / batch_num)
        if epoch_num % 10 == 0:
            saver.save(sess, model_path, global_step=epoch_num)
            with open(loss_ckpt, 'wb') as losses:
                pickle.dump([d_loss_mean, g_loss_mean], losses)
else:
    for num in range(1, 6):
        load_path = 'models/dcgan-{}'.format(num)
        try:
            saver.restore(sess, load_path)
        except:
            continue
        print ("Loaded model: {}".format(load_path))
        os.makedirs(save_path, exist_ok=True)
        for id, captions, z_noise in gen_testing_batch(txt, params):
            gen_image = sess.run(outputs['generator'],
                feed_dict = {
                    input_tensors['t_real_caption'] : captions,
                    input_tensors['t_z'] : z_noise,
                })
            for i in range(sample_num):
                skimage.io.imsave(os.path.join(save_path , 'sample_{}_{}.jpg'.format(id, num)), gen_image[i])
