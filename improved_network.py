import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def deepnn_v2(x_image, output=43):

    padding_pooling = [[0, 0], [0, 1], [0, 1], [0, 0]]

    activation = tf.nn.relu
    if FLAGS.crelu:
        activation = tf.nn.crelu

    weight_decay = tf.contrib.layers.l2_regularizer(scale=0.0001)

    kernel_initialiser = tf.random_uniform_initializer(-0.05, 0.05)

    # First convolutional layer - maps one RGB image to 32 feature maps.
    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=32,
        kernel_initializer=kernel_initialiser,
        kernel_size=[5, 5],
        padding='same',
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv1',
        activation=activation,
    )

    conv1_bn = conv1
    if FLAGS.norm_layer:
        conv1_bn = tf.layers.batch_normalization(conv1)
    # conv1_bn_pad = tf.pad(conv1_bn, padding_pooling, "CONSTANT")
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1_bn,
        pool_size=[3, 3],
        strides=2,
        padding='same',
        name='pool1'
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_initializer=kernel_initialiser,
        kernel_size=[5, 5],
        padding='same',
        activation=activation,
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv2'
    )
    conv2_bn = conv2
    if FLAGS.norm_layer:
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
        kernel_initializer=kernel_initialiser,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=activation,
        use_bias=False,
        kernel_regularizer=weight_decay,
        name='conv3'
    )

    conv3_bn = conv3
    if FLAGS.norm_layer:
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
        kernel_initializer=kernel_initialiser,
        kernel_size=[4, 4],
        padding='same',
        activation=activation,
        kernel_regularizer=weight_decay,
        use_bias=False,
        name='conv4'
    )

    conv4_bn = conv4
    if FLAGS.norm_layer:
        conv4_bn = tf.layers.batch_normalization(conv4)

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
        full_pool = tf.concat([pool1_flat, pool2_flat, pool3_flat, conv4_flat], axis=1)
    else:
        full_pool = conv4_flat

    full_pool = tf.nn.dropout(full_pool , FLAGS.dropout_keep_rate, seed=FLAGS.seed)
    logits = tf.layers.dense(inputs=full_pool,
                             units=output,
                             kernel_regularizer=weight_decay,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01, seed=FLAGS.seed),
                             name='fc1',
                             )
    return logits
