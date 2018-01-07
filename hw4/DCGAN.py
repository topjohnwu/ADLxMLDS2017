import tensorflow as tf
import numpy as np
import ops


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class GAN:
    def __init__(self , params):
        self.params = params

        self.g_bn0 = ops.batch_norm(params, name='g_bn0')
        self.g_bn1 = ops.batch_norm(params, name='g_bn1')
        self.g_bn2 = ops.batch_norm(params, name='g_bn2')
        self.g_bn3 = ops.batch_norm(params, name='g_bn3')

        self.d_bn1 = ops.batch_norm(params, name='d_bn1')
        self.d_bn2 = ops.batch_norm(params, name='d_bn2')
        self.d_bn3 = ops.batch_norm(params, name='d_bn3')
        self.d_bn4 = ops.batch_norm(params, name='d_bn4')

    def build_model(self):

        img_size = self.params['image_size']
        t_real_image = tf.placeholder('float32',[self.params['batch_size'],img_size, img_size, 3 ], name='real_image')
        t_wrong_image = tf.placeholder('float32',[self.params['batch_size'],img_size, img_size, 3 ], name='wrong_image')
        t_real_caption = tf.placeholder('float32', [self.params['batch_size'], self.params['caption_length']], name='real_caption_input')
        t_z = tf.placeholder('float32', [self.params['batch_size'], self.params['z_dim']])


        fake_image = self.generator(t_z, t_real_caption)
        disc_real_image, disc_real_image_logits = self.discriminator(t_real_image, t_real_caption , reuse = False)
        disc_wrong_image, disc_wrong_image_logits = self.discriminator(t_wrong_image, t_real_caption, reuse = True)
        disc_fake_image, disc_fake_image_logits = self.discriminator(fake_image, t_real_caption, reuse = True)


        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.ones_like(disc_fake_image)))
        d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_image_logits, labels=tf.ones_like(disc_real_image)))
        d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_wrong_image_logits, labels=tf.zeros_like(disc_wrong_image)))
        d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits, labels=tf.zeros_like(disc_fake_image)))


        d_loss = d_loss1 + d_loss2 + d_loss3

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        variables = {'d_vars' : d_vars, 'g_vars' : g_vars}

        input_tensors = {
            't_real_image' : t_real_image,
            't_wrong_image' : t_wrong_image,
            't_real_caption' : t_real_caption,
            't_z' : t_z,
        }
        outputs = {'generator' : fake_image}
        loss = {'g_loss' : g_loss, 'd_loss' : d_loss}

        return input_tensors , variables , outputs , loss

    def generator(self , t_z , t_text_embedding):
        with tf.variable_scope("generator") as scope:
            s2 = int(self.params['image_size'] / 2)
            s4 = int(self.params['image_size'] / 4)
            s8 = int(self.params['image_size'] / 8)
            s16 = int(self.params['image_size'] / 16)

            reduced_text_embedding = ops.lrelu(ops.linear(t_text_embedding,self.params['t_dim'], 'g_embedding'))

            g_input = tf.concat([t_z, reduced_text_embedding], 1)
            z_ = ops.linear(g_input, self.params['gf_dim']*8*s16*s16, 'g_h0_lin')
            h0 = tf.reshape(z_, [-1, s16, s16, self.params['gf_dim'] * 8])
            h0 = tf.nn.relu(self.g_bn0(h0))

            h1 = ops.deconv2d(h0, [self.params['batch_size'],
                            s8, s8, self.params['gf_dim']*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = ops.deconv2d(h1, [self.params['batch_size'],
                            s4, s4, self.params['gf_dim']*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = ops.deconv2d(h2, [self.params['batch_size'],
                            s2, s2, self.params['gf_dim']*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = ops.deconv2d(h3, [self.params['batch_size'], self.params['image_size'], self.params['image_size'], 3], name='g_h4')

            return (tf.tanh(h4)/2. + 0.5)

    def discriminator(self , image , t_text_embedding ,reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()

            h0 = ops.lrelu(ops.conv2d(image, self.params['df_dim'], name = 'd_h0_conv')) #32
            h1 = ops.lrelu(self.d_bn1(ops.conv2d(h0, self.params['df_dim']*2, name='d_h1_conv'))) #16
            h2 = ops.lrelu(self.d_bn2(ops.conv2d(h1, self.params['df_dim']*4, name='d_h2_conv'))) #8
            h3 = ops.lrelu(self.d_bn3(ops.conv2d(h2, self.params['df_dim']*8, name='d_h3_conv'))) #4

            reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.params['t_dim'], 'd_embedding'))
            reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
            reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
            tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')

            h3_concat = tf.concat([h3, tiled_embeddings], axis=3, name='h3_concat')
            h3_new = ops.lrelu(self.d_bn4(ops.conv2d(h3_concat, self.params['df_dim']*8, 1, 1, 1, 1, name='d_h3_conv_new'))) #4

            h4 = ops.linear(tf.reshape(h3_new, [self.params['batch_size'], -1]), 1, 'd_h3_lin') # score


            return tf.nn.sigmoid(h4), h4
