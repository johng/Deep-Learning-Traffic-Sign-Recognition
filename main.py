from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import GTSRB as GT
import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline

here = os.path.dirname(__file__)
sys.path.append(here)

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('log-frequency', 100,
                            'Number of steps between logging results to the console and saving summaries. (default: %(default)d)')
tf.app.flags.DEFINE_integer('flush-frequency', 50,
                            'Number of steps between flushing summary results. (default: %(default)d)')
tf.app.flags.DEFINE_integer('save-model-frequency', 5,
                            'Number of steps between model saves. (default: %(default)d)')
tf.app.flags.DEFINE_string('log-dir', '{cwd}/logs/'.format(cwd=os.getcwd()),
                           'Directory where to write event logs and checkpoint. (default: %(default)s)')
# Optimisation hyperparameters
tf.app.flags.DEFINE_integer('max-epochs', 50,
                            'Number of iterations of batch training. (default: %(default)d)')
tf.app.flags.DEFINE_integer('batch-size', 100, 'Number of examples per mini-batch. (default: %(default)d)')
tf.app.flags.DEFINE_float('learning-rate', 0.01, 'Number of examples to run. (default: %(default)d)')
tf.app.flags.DEFINE_float('dropout-keep-rate', 0.7, 'Fraction of connections to keep. (default: %(default)d')
tf.app.flags.DEFINE_integer('early-stop-epochs', 10,
                            'Number of steps without improvement before stopping. (default: %(default)d')


# Graph Options
tf.app.flags.DEFINE_bool('use-profile', False, 'Record trace timeline data')

# Execution environment options
tf.app.flags.DEFINE_float('gpu-memory-fraction', 0.8, 'Fraction of the GPU\'s memory to use')
tf.app.flags.DEFINE_integer('seed', 10, 'Seed')

# Implementation options
tf.app.flags.DEFINE_bool('multi-scale', False, 'Enable multi scale feature. (default: %(default)d')
tf.app.flags.DEFINE_bool('crelu', False, 'Enable CReLU activation. (default: %(default)d')
tf.app.flags.DEFINE_bool('use-augmented-data', False, 'Whether to use pre-generated augmented data on this run')
tf.app.flags.DEFINE_bool('normalise-data', True, 'Whether to normalise the training and test data on a per-image basis')
tf.app.flags.DEFINE_bool('whiten-data', True, 'Whether to \'whiten\' the training and test data on a whole-set basis')

run_log_dir = os.path.join(FLAGS.log_dir, 'exp_bs_{bs}_lr_{lr}_aug_{aug}_nd_{nd}_wd_{wd}'
                           .format(bs=FLAGS.batch_size, lr=FLAGS.learning_rate, aug=FLAGS.use_augmented_data,
                                   nd=FLAGS.normalise_data, wd=FLAGS.whiten_data))

checkpoint_path = os.path.join(run_log_dir, 'model.ckpt')
best_model_path = os.path.join('{cwd}/logs/best'.format(cwd=os.getcwd()), 'model.ckpt')

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)


def deepnn(x_image, output=43):
    padding_pooling = [[0, 0], [0, 1], [0, 1], [0, 0]]

    activation = tf.nn.relu
    if FLAGS.crelu:
        activation = tf.nn.crelu

    weight_decay = tf.contrib.layers.l2_regularizer(scale=0.0001)

    # First convolutional layer - maps one RGB image to 32 feature maps.
    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01, seed=FLAGS.seed),
        kernel_size=[5, 5],
        padding='same',
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv1',
        activation=activation,
    )
    conv1_bn = tf.layers.batch_normalization(conv1)
    # conv1_bn_pad = tf.pad(conv1_bn, padding_pooling, "CONSTANT")
    pool1 = tf.layers.average_pooling2d(
        inputs=conv1_bn,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool1'
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01, seed=FLAGS.seed),
        kernel_size=[5, 5],
        padding='same',
        activation=activation,
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv2'
    )
    conv2_bn = tf.layers.batch_normalization(conv2)
    # conv2_bn_pad = tf.pad(conv2_bn, padding_pooling, "CONSTANT")
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2_bn,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool2'
    )

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01, seed=FLAGS.seed),
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=activation,
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv3'
    )
    conv3_bn = tf.layers.batch_normalization(conv3)
    conv3bn_pad = tf.pad(conv3_bn, padding_pooling, "CONSTANT")
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3_bn,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool3'
    )

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=64,
        kernel_size=[4, 4],
        padding='same',
        activation=activation,
        kernel_regularizer=weight_decay,
        use_bias=False,
        name='conv4'
    )
    conv4_bn = (tf.layers.batch_normalization(conv4))

    pool1_multiscale = tf.layers.max_pooling2d(
        inputs=conv4_bn,
        pool_size=[3, 3],
        strides=1,
        padding='same',
        name='pool1_multiscale'
    )
    pool2_multiscale = tf.layers.max_pooling2d(
        inputs=conv2_bn,
        pool_size=[3, 3],
        strides=(4, 2),
        padding='same',
        name='pool2_multiscale'
    )
    pool3_multiscale = tf.layers.max_pooling2d(
        inputs=conv3_bn,
        pool_size=[3, 3],
        strides=(2, 2),
        padding='same',
        name='pool3_multiscale'

    )
    # Multi-Scale features - fast forward earlier layer results
    pool1_flat = tf.contrib.layers.flatten(pool1_multiscale)
    pool2_flat = tf.contrib.layers.flatten(pool2_multiscale)
    pool3_flat = tf.contrib.layers.flatten(pool3_multiscale)
    conv4_flat = tf.contrib.layers.flatten(conv4_bn)

    if FLAGS.multi_scale:
        full_pool = tf.nn.dropout(tf.concat([pool1_flat, pool2_flat, pool3_flat, conv4_flat], axis=1),
                                  FLAGS.dropout_keep_rate, seed=FLAGS.seed)
    else:
        full_pool = conv4_flat

    logits = tf.layers.dense(inputs=full_pool,
                             units=output,
                             kernel_regularizer=weight_decay,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01, seed=FLAGS.seed),
                             name='fc1',
                             )
    return logits


def main(_):
    tf.reset_default_graph()
    tf.set_random_seed(FLAGS.seed)
    gtsrb = GT.GTSRB(batch_size=FLAGS.batch_size, use_augmented_data=FLAGS.use_augmented_data,
                     normalise_data=FLAGS.normalise_data, whiten_data=FLAGS.whiten_data, seed=FLAGS.seed)
    augment = tf.placeholder(tf.bool)

    # Build the graph for the deep net
    with tf.name_scope('inputs'):
        y_ = tf.placeholder(tf.float32, [None, gtsrb.OUTPUT])
        x = tf.placeholder(tf.float32, [None, gtsrb.WIDTH * gtsrb.HEIGHT * gtsrb.CHANNELS])
        x_image = tf.reshape(x, [-1, gtsrb.WIDTH, gtsrb.HEIGHT, gtsrb.CHANNELS])

        global_epoch = tf.placeholder(tf.int32)

    with tf.name_scope('model'):
        y_conv = deepnn(x_image)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    correct_prediction = tf.equal(tf.argmax(y_conv, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    class_counts = tf.count_nonzero(y_, 0)
    correct_per_class = tf.unsorted_segment_sum(data=tf.to_float(correct_prediction), segment_ids=tf.argmax(y_, axis=1),
                                                num_segments=43)

    global_step = tf.Variable(0, trainable=False)  # this will be incremented automatically by tensorflow
    decay_steps = 30  # decay the learning rate every 1000 steps
    decay_rate = 0.9  # the base of our exponential for the decay
    decayed_learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_epoch,
                                                       decay_steps, decay_rate, staircase=False)

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
    best_saver = tf.train.Saver(max_to_keep=1)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_writer = tf.summary.FileWriter(run_log_dir + "_train", sess.graph)
        validation_writer = tf.summary.FileWriter(run_log_dir + "_validation", sess.graph)

        sess.run(tf.global_variables_initializer())
        options = None
        run_metadata = None
        if FLAGS.use_profile:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

        best_accuracy = 0
        steps_since_last_improvement = 0
        # Batch generator used for validation
        # Training and validation
        for step in range(FLAGS.max_epochs):
            # Batch generator used for training in each epoch
            train_batch_generator = gtsrb.batch_generator('train', batch_size=FLAGS.batch_size, limit=True)
            # rotated_training_images = sess.run([rotation], feed_dict={x_image: trainImages})
            for (trainImages, trainLabels) in train_batch_generator:
                _, train_summary_str = sess.run([train_step, train_summary],
                                                feed_dict={x_image: trainImages, y_: trainLabels, augment: True,
                                                           global_epoch: step},
                                                options=options, run_metadata=run_metadata)

            validation_batch_generator = gtsrb.batch_generator('test', batch_size=FLAGS.batch_size, limit=True,
                                                               fraction=1.0)
            # Validation: Monitoring accuracy using validation set
            total_validation_accuracy = 0
            validation_batches = 0
            for (testImages, testLabels) in validation_batch_generator:
                validation_accuracy, validation_summary_str = sess.run([accuracy, validation_summary],
                                                                       feed_dict={x_image: testImages, y_: testLabels,
                                                                                  augment: False})
                total_validation_accuracy += validation_accuracy
                validation_batches += 1
            validation_accuracy = total_validation_accuracy / validation_batches
            print('epoch %02d, accuracy on validation set : %.3f' % (step + 1, validation_accuracy))
            if validation_accuracy >= best_accuracy:
                best_saver.save(sess, best_model_path)
                best_accuracy = validation_accuracy
                steps_since_last_improvement = 0
            else:
                steps_since_last_improvement += 1
            train_writer.add_summary(train_summary_str, step)
            validation_writer.add_summary(validation_summary_str, step)

            # Save the model checkpoint periodically.
            if (step + 1) % FLAGS.save_model_frequency == 0 or (step + 1) == FLAGS.max_epochs:
                saver.save(sess, checkpoint_path, global_step=step)

            if (step + 1) % FLAGS.flush_frequency == 0:
                train_writer.flush()
                validation_writer.flush()
            if steps_since_last_improvement >= FLAGS.early_stop_epochs:
                print('Stopping early')
                break

        # Resetting the internal batch indexes
        evaluated_images = 0
        test_accuracy = 0
        batch_count = 0
        test_class_counts = np.zeros(43)
        test_correct_per_class = np.zeros(43)

        gtsrb.reset()
        best_saver.restore(sess, best_model_path)
        test_batch_generator = gtsrb.batch_generator('test', batch_size=FLAGS.batch_size, limit=True)
        for (testImages, testLabels) in test_batch_generator:
            test_accuracy_temp, test_class_counts_temp, test_correct_per_class_temp = sess.run(
                [accuracy, class_counts, correct_per_class],
                feed_dict={x_image: testImages, y_: testLabels,
                           augment: False})

            batch_count += 1
            test_accuracy += test_accuracy_temp
            test_class_counts = np.add(test_class_counts, test_class_counts_temp)
            test_correct_per_class = np.add(test_correct_per_class, test_correct_per_class_temp)
            evaluated_images += len(testLabels)

        test_accuracy = test_accuracy / batch_count
        test_accuracy_per_class = test_correct_per_class / test_class_counts
        print('test set: accuracy on test set: %.4f' % test_accuracy)
        print('test set: check accuracy: {:.4f}'.format(test_correct_per_class.sum() / test_class_counts.sum()))
        print('test set: accuracy on speed limits: {:.4f}'.format(
            test_accuracy_per_class[GT.GTSRB.speed_limit_classes].sum() / len(GT.GTSRB.speed_limit_classes)))
        print('test set: accuracy on prohibitory: {:.4f}'.format(
            test_accuracy_per_class[GT.GTSRB.prohibitory_classes].sum() / len(GT.GTSRB.prohibitory_classes)))
        print('test set: accuracy on derestriction: {:.4f}'.format(
            test_accuracy_per_class[GT.GTSRB.derestriction_classes].sum() / len(GT.GTSRB.derestriction_classes)))
        print('test set: accuracy on mandatory: {:.4f}'.format(
            test_accuracy_per_class[GT.GTSRB.mandatory_classes].sum() / len(GT.GTSRB.mandatory_classes)))
        print('test set: accuracy on danger: {:.4f}'.format(
            test_accuracy_per_class[GT.GTSRB.danger_classes].sum() / len(GT.GTSRB.danger_classes)))
        print('test set: accuracy on unique: {:.4f}'.format(
            test_accuracy_per_class[GT.GTSRB.unique_classes].sum() / len(GT.GTSRB.unique_classes)))
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
