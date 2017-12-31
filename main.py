from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import GTSRB as GT
import tensorflow as tf
import random

here = os.path.dirname(__file__)
sys.path.append(here)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('log-frequency', 100,
                            'Number of steps between logging results to the console and saving summaries. (default: %(default)d)')
tf.app.flags.DEFINE_integer('flush-frequency', 50,
                            'Number of steps between flushing summary results. (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model-frequency', 100,
                            'Number of steps between model saves. (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('max-steps', 10000,
                            'Number of mini-batches to train on. (default: %(default)d)')
tf.app.flags.DEFINE_integer('batch-size', 100, 'Number of examples per mini-batch. (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 0.01, 'Number of examples to run. (default: %(default)d)')

# Graph Options
tf.app.flags.DEFINE_bool('data-augment', True, 'Add randomized rotation and flipping to training data')

run_log_dir = os.path.join(FLAGS.log_dir, 'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size, lr=FLAGS.learning_rate))

checkpoint_path = os.path.join(run_log_dir, 'model.ckpt')

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)


def deepnn(x_image, output=43):
    padding_pooling = [[0, 0], [0, 1], [0, 1], [0, 0]]

    weight_decay = tf.contrib.layers.l2_regularizer(scale=0.0001)

    # First convolutional layer - maps one RGB image to 32 feature maps.
    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        kernel_size=[5, 5],
        padding='same',
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv1'
    )
    conv1_bn = tf.nn.crelu(tf.layers.batch_normalization(conv1))
    conv1_bn_pad = tf.pad(conv1_bn, padding_pooling, "CONSTANT")
    pool1 = tf.layers.average_pooling2d(
        inputs=conv1_bn_pad,
        pool_size=[3, 3],
        strides=2,
        padding='valid',
        name='pool1'
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.crelu,
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv2'
    )
    conv2_bn = tf.nn.crelu(tf.layers.batch_normalization(conv2))
    conv2_bn_pad = tf.pad(conv2_bn, padding_pooling, "CONSTANT")
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2_bn_pad,
        pool_size=[3, 3],
        strides=2,
        padding='valid',
        name='pool2'
    )

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.crelu,
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv3'
    )
    conv3_bn = tf.nn.crelu(tf.layers.batch_normalization(conv3))
    conv3bn_pad = tf.pad(conv3_bn, padding_pooling, "CONSTANT")
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3bn_pad,
        pool_size=[3, 3],
        strides=2,
        padding='valid',
        name='pool3'
    )
    pool_drop = tf.nn.dropout(pool3, 0.7)

    conv4 = tf.layers.conv2d(
        inputs=pool_drop,
        filters=64,
        kernel_size=[4, 4],
        padding='valid',
        activation=tf.nn.relu,
        kernel_regularizer=weight_decay,
        use_bias=False,
        name='conv4'
    )
    conv4_bn = tf.nn.relu(tf.layers.batch_normalization(conv4))

    pool4_flat = tf.reshape(conv4_bn, [-1, 1 * 1 * 64], name='conv4_bn_flattened')


    logits = tf.layers.dense(inputs=pool4_flat,
                             units=output,
                             kernel_regularizer=weight_decay,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                             name='fc1',
                             )
    return logits


def main(_):
    tf.reset_default_graph()

    gtsrb = GT.gtsrb(batch_size=FLAGS.batch_size)

    augment = tf.placeholder(tf.bool)
    # Build the graph for the deep net
    with tf.name_scope('inputs'):

        x = tf.placeholder(tf.float32, [None, gtsrb.WIDTH * gtsrb.HEIGHT * gtsrb.CHANNELS])
        x_image = tf.reshape(x, [-1, gtsrb.WIDTH, gtsrb.HEIGHT, gtsrb.CHANNELS])
        transform = tf.map_fn(lambda v: tf.image.random_flip_up_down(v), x_image)

        # Transformations as listed in http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6033589
        # random rotation -15,15 degrees
        def random_rotate(): return tf.map_fn(lambda img: tf.contrib.image.rotate(img, random.uniform(-0.26, 0.26)),
                                              x_image)

        # random translation -2,2 pixels
        def random_translate():
            return tf.map_fn(lambda img: tf.contrib.image.transform(img, [1, 0, random.randint(-2, 2), 0, 1,
                                                                          random.randint(-2, 2), 0, 0]), x_image)
        #We can also flip images whose class is invariant to flips
        #Can also flip & reclassify where applicable

        #x_image = tf.cond(augment, random_rotate, lambda: tf.identity(x_image))
        # x_image = tf.cond(augment, random_translate, lambda: tf.identity(x_image))
        x_image = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_image)
        x_image = tf.map_fn(lambda img: tf.image.rgb_to_hsv(img), x_image)

        y_ = tf.placeholder(tf.float32, [None, gtsrb.OUTPUT])

    with tf.name_scope('model'):
        y_conv = deepnn(x_image)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    global_step = tf.Variable(0, trainable=False)  # this will be incremented automatically by tensorflow
    decay_steps = 1000  # decay the learning rate every 1000 steps
    decay_rate = 0.9  # the base of our exponential for the decay
    decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                                       decay_steps, decay_rate, staircase=True)

    # We need to update the dependencies of the minimization op so that it all ops in the `UPDATE_OPS`
    # are added as a dependency, this ensures that we update the mean and variance of the batch normalisation
    # layers
    # See https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization for more
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.MomentumOptimizer(decayed_learning_rate, 0.9).minimize(cross_entropy,
                                                                                     global_step=global_step)

    loss_summary = tf.summary.scalar("Loss", cross_entropy)
    accuracy_summary = tf.summary.scalar("Accuracy", accuracy)
    learning_rate_summary = tf.summary.scalar("Learning Rate", decayed_learning_rate)
    img_summary = tf.summary.image('input images', x_image)

    train_summary = tf.summary.merge([loss_summary, accuracy_summary, learning_rate_summary, img_summary])
    validation_summary = tf.summary.merge([loss_summary, accuracy_summary])

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        validation_writer = tf.summary.FileWriter(run_log_dir + "_validation", sess.graph)

        sess.run(tf.global_variables_initializer())

        # Training and validation
        for step in range(FLAGS.max_steps):
            (trainImages, trainLabels) = gtsrb.get_train_batch()
            (testImages, testLabels) = gtsrb.get_test_batch()
            # rotated_training_images = sess.run([rotation], feed_dict={x_image: trainImages})
            _, train_summary_str = sess.run([train_step, train_summary],
                                            feed_dict={x_image: trainImages, y_: trainLabels, augment: True})

            # Validation: Monitoring accuracy using validation set
            if (step + 1) % FLAGS.log_frequency == 0:
                validation_accuracy, validation_summary_str = sess.run([accuracy, validation_summary],
                                                                       feed_dict={x_image: testImages, y_: testLabels,
                                                                                  augment: False})
                print('step {}, accuracy on validation set : {}'.format(step + 1, validation_accuracy))
                train_writer.add_summary(train_summary_str, step)
                validation_writer.add_summary(validation_summary_str, step)

            # Save the model checkpoint periodically.
            if (step + 1) % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_steps:
                saver.save(sess, checkpoint_path, global_step=step)

            if (step + 1) % FLAGS.flush_frequency == 0:
                train_writer.flush()
                validation_writer.flush()

        # Resetting the internal batch indexes
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0

        gtsrb.reset()

        while evaluated_images != gtsrb.nTestSamples:
            # Don't loop back when we reach the end of the test set
            (testImages, testLabels) = gtsrb.get_test_batch(allow_smaller_batches=True)
            test_accuracy_temp = sess.run(accuracy, feed_dict={x_image: testImages, y_: testLabels, augment: False})

            batch_count += 1
            test_accuracy += test_accuracy_temp
            evaluated_images += testLabels.shape[0]

        test_accuracy = test_accuracy / batch_count
        print('test set: accuracy on test set: %0.3f' % test_accuracy)
        print('model saved to ' + checkpoint_path)

        train_writer.close()
        validation_writer.close()


if __name__ == '__main__':
    tf.app.run()
