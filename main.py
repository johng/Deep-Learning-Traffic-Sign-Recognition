from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import sqrt
import sys
import os
import GTSRB as GT
import tensorflow as tf
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
tf.app.flags.DEFINE_bool('data-augment', True, 'Add randomized rotation and flipping to training data')
tf.app.flags.DEFINE_bool('use-profile', False, 'Record trace timeline data')

# Execution environment options
tf.app.flags.DEFINE_float('gpu-memory-fraction', 0.8, 'Fraction of the GPU\'s memory to use')
tf.app.flags.DEFINE_bool('generate-augmented-data', False, 'Whether to generate augmented data on this run')
tf.app.flags.DEFINE_bool('use-augmented-data', True, 'Whether to use pre-existing augmented data on this run')

run_log_dir = os.path.join(FLAGS.log_dir, 'exp_bs_{bs}_lr_{lr}'.format(bs=FLAGS.batch_size, lr=FLAGS.learning_rate))

checkpoint_path = os.path.join(run_log_dir, 'model.ckpt')
best_model_path = os.path.join('{cwd}/logs/best'.format(cwd=os.getcwd()), 'model.ckpt')

# limit the process memory to a third of the total gpu memory
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)


def deepnn(x_image, output=43):
    conv_padding = [[0, 0], [2,2], [2,2], [0,0]]
    pool_padding = [ [0,0], [0,1], [0,1], [0,0]]

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
    conv1_bn = tf.nn.relu(tf.layers.batch_normalization(conv1))
    conv1_bn_pad = tf.pad(conv1_bn, pool_padding, "CONSTANT")
    pool1 = tf.layers.average_pooling2d(
        inputs=conv1_bn,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool1',
        data_format='channels_first'
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv2',
        data_format='channels_first'
    )
    conv2_bn = tf.nn.relu(tf.layers.batch_normalization(conv2))
    conv2_bn_pad = tf.pad(conv2_bn, pool_padding, "CONSTANT")
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2_bn,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool2',
        data_format='channels_first'
    )

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu,
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv3',
        data_format='channels_first'
    )
    conv3_bn = tf.nn.relu(tf.layers.batch_normalization(conv3))
    conv3bn_pad = tf.pad(conv3_bn, pool_padding, "CONSTANT")
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3_bn,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool3',
        data_format='channels_first'
    )

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=64,
        kernel_size=[4, 4],
        padding='valid',
        activation=tf.nn.relu,
        kernel_regularizer=weight_decay,
        use_bias=False,
        name='conv4'
    )


    conv4_bn = tf.nn.relu(tf.layers.batch_normalization(conv4))

    # Multi-Scale features - fast forward earlier layer results
    pool1_flat = tf.contrib.layers.flatten(pool1)
    pool2_flat = tf.contrib.layers.flatten(pool2)
    pool3_flat = tf.contrib.layers.flatten(pool3)
    conv4_flat = tf.contrib.layers.flatten(conv4_bn)

    full_pool = tf.nn.dropout(tf.concat([pool1_flat, pool2_flat, pool3_flat, conv4_flat], axis=1),
                              FLAGS.dropout_keep_rate)
    logits = tf.layers.dense(inputs=conv4_flat,
                             units=output,
                             kernel_regularizer=weight_decay,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01),
                             name='fc1',
                             )
    return logits


def put_kernels_on_grid (kernel, grid_Y, grid_X, pad = 1):

    '''Visualize conv. features as an image (mostly for the 1st layer).
    Place kernel into a grid, with some paddings between adjacent filters.

    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                           User is responsible of how to break into two multiples.
      pad:               number of black pixels around each filter (between them)

    Return:
      Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
    '''

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)

    kernel1 = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x1 = tf.pad(kernel1, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel1.get_shape()[0] + 2 * pad
    X = kernel1.get_shape()[1] + 2 * pad

    channels = kernel1.get_shape()[2]

    # put NumKernels to the 1st dimension
    x2 = tf.transpose(x1, (3, 0, 1, 2))
    # organize grid on Y axis
    x3 = tf.reshape(x2, tf.stack([grid_X, Y * grid_Y, X, channels])) #3

    # switch X and Y axes
    x4 = tf.transpose(x3, (0, 2, 1, 3))
    # organize grid on X axis
    x5 = tf.reshape(x4, tf.stack([1, X * grid_X, Y * grid_Y, channels])) #3

    # back to normal order (not combining with the next step for clarity)
    x6 = tf.transpose(x5, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x7 = tf.transpose(x6, (3, 0, 1, 2))

    # scale to [0, 255] and convert to uint8
    return tf.image.convert_image_dtype(x7, dtype = tf.uint8)
def main(_):
    tf.reset_default_graph()
    gtsrb = GT.gtsrb(batch_size=FLAGS.batch_size, use_extended=FLAGS.use_augmented_data,
                     generate_extended=FLAGS.generate_augmented_data)
    augment = tf.placeholder(tf.bool)
    # Build the graph for the deep net
    with tf.name_scope('inputs'):
        y_ = tf.placeholder(tf.float32, [None, gtsrb.OUTPUT])
        x = tf.placeholder(tf.float32, [None, gtsrb.WIDTH * gtsrb.HEIGHT * gtsrb.CHANNELS])
        x_image = tf.reshape(x, [-1, gtsrb.WIDTH, gtsrb.HEIGHT, gtsrb.CHANNELS])
        x_image = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_image)
        global_epoch = tf.placeholder(tf.int32)

    with tf.name_scope('model'):
        y_conv = deepnn(x_image)

    with tf.variable_scope('conv1', reuse=True):
        kernel1 = tf.get_variable('kernel')
        grid1 = put_kernels_on_grid(kernel1, 8,4)
        conv1_features = tf.summary.image('conv1/kernels', grid1, max_outputs=1)

    with tf.variable_scope('conv2', reuse=True):
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv2')
        print(vars)
        kernel2 = tf.get_variable('kernel')
        reduced= tf.reduce_mean(kernel2,axis=2,keep_dims=True)
        grid2 = put_kernels_on_grid(reduced, 8 , 4)
        # [1,56,28,64]
        conv2_features = tf.summary.image('conv2/kernels', grid2, max_outputs=1)


    conv_features = tf.summary.merge([ conv1_features, conv2_features])


    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    global_step = tf.Variable(0, trainable=False)  # this will be incremented automatically by tensorflow
    decay_steps = 10  # decay the learning rate every 1000 steps
    decay_rate = 0.95  # the base of our exponential for the decay
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

    train_summary = tf.summary.merge([loss_summary, accuracy_summary, learning_rate_summary, img_summary, conv_features])
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
            print('epoch {}, accuracy on validation set : {}'.format(step + 1, validation_accuracy))
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

        gtsrb.reset()
        best_saver.restore(sess, best_model_path)
        test_batch_generator = gtsrb.batch_generator('test', batch_size=FLAGS.batch_size, limit=True)
        for (testImages, testLabels) in test_batch_generator:
            test_accuracy_temp = sess.run(accuracy, feed_dict={x_image: testImages, y_: testLabels, augment: False})

            batch_count += 1
            test_accuracy += test_accuracy_temp
            evaluated_images += len(testLabels)

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
