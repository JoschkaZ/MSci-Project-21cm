
#%% IMPORTS
from __future__ import division, print_function, absolute_import
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler

logs_path = '/tmp/tensorflow_logs/example11/'

def scale_slices(slices):

    #max = 40.77593
    #min = 0
    #for slice in slices:
    #scaler = MinMaxScaler()
    #print(scaler.fit(slices))
    scaled = np.interp(slices, (slices.min(), slices.max()), (-1, +1))

    #print(slices[0])
    #print(scaled[0])


    return scaled


slices = pickle.load(open(r"C:\\Outputs\\slices.pkl", "rb"))
slices = np.array(slices)
print(len(slices))
print(np.shape(slices))

slices = scale_slices(slices)
print(len(slices))
print(np.shape(slices))

#%% GET MNIST SHAPE
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
batch_size=5
batch_x, y = mnist.train.next_batch(batch_size)
print(np.shape(batch_x))

# %% HYPER PARAMETERS
noise_dim = 200


#%% Generator Network
# Input: Noise, Output: Image
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.

        x = tf.layers.dense(x, units=4 * 4 * 1280)
        x = tf.nn.tanh(x)
        # New shape: (20480)

        x = tf.reshape(x, shape=[-1, 4, 4, 1280])
        # New shape: (batch, 4, 4, 1280)

        x = tf.layers.conv2d_transpose(x, 640, 2, strides=2)
        # New shape: (batch, 8, 8, 640)

        x = tf.layers.conv2d_transpose(x, 320, 2, strides=2)
        # New shape: (batch, 16, 16, 320)

        x = tf.layers.conv2d_transpose(x, 160, 2, strides=2)
        # New shape: (batch, 32, 32, 160)

        x = tf.layers.conv2d_transpose(x, 80, 2, strides=2)
        # New shape: (batch, 64, 64, 80)

        x = tf.layers.conv2d_transpose(x, 40, 2, strides=2)
        # New shape: (batch, 128, 128, 40)

        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        # New shape: (batch, 256, 256, 1)

        x = tf.nn.sigmoid(x)
        return x


#%% Discriminator Network
# Input: Image, Output: Prediction Real/Fake Image
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):

        # Input shape: (batch, 256, 256, 1)

        x = tf.layers.conv2d(x, 6, 5)
        x = tf.nn.tanh(x)
        # New shape: (batch, 252, 252, 6)
        x = tf.layers.average_pooling2d(x, 2, 2)
        # New shape: (batch, 126, 126, 6)

        x = tf.layers.conv2d(x, 12, 5)
        x = tf.nn.tanh(x)
        # New shape: (batch, 122, 122, 12)
        x = tf.layers.average_pooling2d(x, 2, 2)
        # New shape: (batch, 61, 61, 12)

        x = tf.layers.conv2d(x, 24, 6)
        x = tf.nn.tanh(x)
        # New shape: (batch, 56, 56, 24)
        x = tf.layers.average_pooling2d(x, 2, 2)
        # New shape: (batch, 28, 28, 24)

        x = tf.layers.conv2d(x, 48, 5)
        x = tf.nn.tanh(x)
        # New shape: (batch, 24, 24, 48)
        x = tf.layers.average_pooling2d(x, 2, 2)
        # New shape: (batch, 12, 12, 48)

        x = tf.layers.conv2d(x, 96, 5)
        x = tf.nn.tanh(x)
        # New shape: (batch, 8, 8, 96)
        x = tf.layers.average_pooling2d(x, 2, 2)
        # New shape: (batch, 4, 4, 96)

        x = tf.layers.conv2d(x, 192, 4)
        x = tf.nn.tanh(x)
        # New shape: (batch, 1, 1, 192)

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        x = tf.layers.dense(x, 2)
        # New shape (2) (real or fake)

    return x



# %% BUILD NETWORKS
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])

# Build Generator Network
gen_sample = generator(noise_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)
disc_concat = tf.concat([disc_real, disc_fake], axis=0)

# Build the stacked generator/discriminator
stacked_gan = discriminator(gen_sample, reuse=True)

# Build Targets (real or fake images)
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

# Build Loss
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_concat, labels=disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=gen_target))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.0001)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

#summaries
tf.summary.scalar("disc_loss", disc_loss)
tf.summary.scalar("gen_loss", gen_loss)
merged_summary_op = tf.summary.merge_all()


# Start training
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


    batch_size = 20
    num_steps = int(1.*len(slices) / batch_size)
    epochs = 100

    for epoch in range(epochs):
        print('EPOCH ',epoch)
        for i in range(num_steps):

            # Prepare Input Data
            # Get the next batch of MNIST data (only images are needed, not labels)
            batch_x = slices[i*batch_size:(i+1)*batch_size]
            #print('batchshape ', np.shape(batch_x))
            batch_x = np.reshape(batch_x, newshape=[-1, 256, 256, 1])
            # Generate noise to feed to the generator
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

            # Prepare Targets (Real image: 1, Fake image: 0)
            # The first half of data fed to the generator are real images,
            # the other half are fake images (coming from the generator).
            batch_disc_y = np.concatenate(
                [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
            # Generator tries to fool the discriminator, thus targets are 1.
            batch_gen_y = np.ones([batch_size])

            # Training
            feed_dict = {real_image_input: batch_x, noise_input: z,
                         disc_target: batch_disc_y, gen_target: batch_gen_y}
            #_, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
            #                        feed_dict=feed_dict)

            _, _, summary = sess.run([train_gen, train_disc, merged_summary_op],
                                    feed_dict=feed_dict)


            summary_writer.add_summary(summary, epoch * num_steps + i)


            if i % 100 == 0 or i == 1:
                #print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
                print(summary)

    # Generate images from noise, using the generator network.
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[4, noise_dim])
        g = sess.run(gen_sample, feed_dict={noise_input: z})
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(256, 256, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
#plt.waitforbuttonpress()
plt.savefig('test.png')
print('Figure Saved!')


print("Run the command line:\n" \
      "--> tensorboard --logdir=/tmp/tensorflow_logs " \
"\nThen open http://0.0.0.0:6006/ into your web browser")
