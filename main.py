from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import GTSRB as GT
import tensorflow as tf
import random
from tensorflow.python.client import timeline

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
tf.app.flags.DEFINE_integer('batch-size', 50, 'Number of examples per mini-batch. (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 0.01, 'Number of examples to run. (default: %(default)d)')
tf.app.flags.DEFINE_float('dropout-keep-rate', 0.90, 'Fraction of connections to keep. (default: %(default)d')

# Graph Options
tf.app.flags.DEFINE_bool('data-augment', True, 'Add randomized rotation and flipping to training data')

tf.app.flags.DEFINE_bool('use-profile', False, 'Record trace timeline data')

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
    conv1_bn = tf.nn.crelu(tf.layers.batch_normalization(conv1, fused=True))
    conv1_bn_pad = tf.pad(conv1_bn, padding_pooling, "CONSTANT")
    pool1 = tf.layers.average_pooling2d(
        inputs=conv1_bn_pad,
        pool_size=[3, 3],
        strides=2,
        padding='valid',
        name='pool1'
    )

    pool1_drop = tf.nn.dropout(pool1, FLAGS.dropout_keep_rate)

    conv2 = tf.layers.conv2d(
        inputs=pool1_drop,
        filters=32,
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.crelu,
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv2'
    )
    conv2_bn = tf.nn.crelu(tf.layers.batch_normalization(conv2, fused=True))
    conv2_bn_pad = tf.pad(conv2_bn, padding_pooling, "CONSTANT")
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2_bn_pad,
        pool_size=[3, 3],
        strides=2,
        padding='valid',
        name='pool2'
    )
    pool2_drop = tf.nn.dropout(pool2, FLAGS.dropout_keep_rate)


    conv3 = tf.layers.conv2d(
        inputs=pool2_drop,
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.crelu,
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv3'
    )
    conv3_bn = tf.nn.crelu(tf.layers.batch_normalization(conv3, fused=True))
    conv3bn_pad = tf.pad(conv3_bn, padding_pooling, "CONSTANT")
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3bn_pad,
        pool_size=[3, 3],
        strides=2,
        padding='valid',
        name='pool3'
    )
    pool3_drop = tf.nn.dropout(pool3, FLAGS.dropout_keep_rate)


    conv4 = tf.layers.conv2d(
        inputs=pool3_drop,
        filters=64,
        kernel_size=[4, 4],
        padding='valid',
        activation=tf.nn.relu,
        kernel_regularizer=weight_decay,
        use_bias=False,
        name='conv4'
    )
    conv4_bn = tf.nn.relu(tf.layers.batch_normalization(conv4, fused=True))
    pool4_drop = tf.nn.dropout(conv4_bn, FLAGS.dropout_keep_rate)

    pool1_flat = tf.contrib.layers.flatten(pool1_drop)
    pool2_flat = tf.contrib.layers.flatten(pool2_drop)
    pool3_flat = tf.contrib.layers.flatten(pool3_drop)
    pool4_flat = tf.contrib.layers.flatten(pool4_drop)

    full_pool = tf.concat([pool1_flat, pool2_flat, pool3_flat, pool4_flat], axis=1)
    logits = tf.layers.dense(inputs=full_pool,
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
        y_ = tf.placeholder(tf.float32, [None, gtsrb.OUTPUT])
        x = tf.placeholder(tf.float32, [None, gtsrb.WIDTH * gtsrb.HEIGHT * gtsrb.CHANNELS])
        x_image = tf.reshape(x, [-1, gtsrb.WIDTH, gtsrb.HEIGHT, gtsrb.CHANNELS])
        transform = tf.map_fn(lambda v: tf.image.random_flip_up_down(v), x_image)

        flip_invariant_classes = tf.constant([17, 12, 13, 15, 35])
        # Transformations as listed in http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6033589
        # random rotation -15,15 degrees
        def random_rotate(): return tf.map_fn(lambda img: tf.contrib.image.rotate(img, random.uniform(-0.26, 0.26)),
                                              x_image)

        # random translation -2,2 pixels
        def random_translate():
            return tf.map_fn(lambda img: tf.contrib.image.transform(img, [1, 0, random.randint(-2, 2), 0, 1,
                                                                          random.randint(-2, 2), 0, 0]), x_image)
        def rotate_and_extend():
            x_image_rotated = random_rotate()
            x_image_extended = tf.concat([x_image, x_image_rotated],0)
            y_extended = tf.concat([y_, y_],0)
            return x_image_extended, y_extended

        def flip_valid(image, label):
            label_idx = tf.argmax(label, output_type=tf.int32)
            is_valid = tf.foldl(lambda a, b: a | b, tf.equal(label_idx, flip_invariant_classes))
            return tf.cond(is_valid, lambda: tf.image.flip_left_right(image), lambda: tf.identity(image))

        def flip_valid_and_extend():
            x_images = tf.unstack(x_image, num=FLAGS.batch_size)
            labels = tf.unstack(y_, num=FLAGS.batch_size)
            flipped = []
            for idx, img in enumerate(x_images):
                flipped.append(flip_valid(img, labels[idx]))
            x_image_flipped = tf.stack(flipped)
            x_image_extended = tf.concat([x_image, x_image_flipped], 0)
            y_extended = tf.concat([y_, y_], 0)
            return x_image_extended, y_extended


        #We can also flip images whose class is invariant to flips
        #Can also flip & reclassify where applicable
        x_image, y_ = tf.cond(augment, flip_valid_and_extend, lambda: (tf.identity(x_image), tf.identity(y_)))
        x_image, y_ = tf.cond(augment, rotate_and_extend, lambda: (tf.identity(x_image), tf.identity(y_)))
        #x_image = tf.cond(augment, random_translate, lambda: tf.identity(x_image))
        x_image = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_image)
        x_image = tf.map_fn(lambda img: tf.image.rgb_to_hsv(img), x_image)


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
        options = None
        run_metadata = None
        if FLAGS.use_profile:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        # Training and validation
        for step in range(FLAGS.max_steps):
            (trainImages, trainLabels) = gtsrb.get_train_batch()
            (testImages, testLabels) = gtsrb.get_test_batch()
            # rotated_training_images = sess.run([rotation], feed_dict={x_image: trainImages})
            _, train_summary_str = sess.run([train_step, train_summary],
                                            feed_dict={x_image: trainImages, y_: trainLabels, augment: True},
                                            options=options, run_metadata=run_metadata)

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
        if FLAGS.use_profile:
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('timeline_01.json', 'w') as f:
                f.write(chrome_trace)

if __name__ == '__main__':
    tf.app.run()
