import numpy as np
import tensorflow as tf


class GTSRB:
    WIDTH = 32
    HEIGHT = 32
    CHANNELS = 3
    OUTPUT = 43
    num_test_items = -1

    def __init__(self, batch_size=128, use_augmented_data=False, normalise_data=True, whiten_data=True):
        dataset = np.load('gtsrb_dataset.npz')

        self.train_data = dataset['X_train']
        self.train_labels = dataset['y_train']
        print('Original data:   {}'.format(self.train_data.shape))
        print('Original labels: {}'.format(self.train_labels.shape))
        self.test_data = dataset['X_test']
        self.test_labels = dataset['y_test']

        if use_augmented_data:
            augmented_dataset = np.load('extended_dataset.npz')
            self.train_data = np.append(self.train_data, augmented_dataset['images'], axis=0)
            self.train_labels = np.append(self.train_labels, augmented_dataset['labels'], axis=0)
            print('All data:   {}'.format(self.train_data.shape))
            print('All labels: {}'.format(self.train_labels.shape))

        if normalise_data:
            self.train_data = self.normalise_images(self.train_data)
            self.test_data = self.normalise_images(self.test_data)

        if whiten_data:
            self.train_data = self.whiten_images(self.train_data)
            for i in range(0, 3):
                self.test_data[:][:][:][i] = (self.test_data[:][:][:][i] - self.means[i]) / self.stddevs[i]

        self.num_train_items = len(self.train_labels)
        self.num_test_items = len(self.test_labels)

        self.batch_size = batch_size

        self.permutation_train = np.random.permutation(self.num_train_items)
        self.permutation_test = np.random.permutation(self.num_test_items)

        self.current_idx_test = 0
        self.current_idx_train = 0

    def normalise_images(self, images):
        sess = tf.Session()
        x_image = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, self.CHANNELS])
        normalize = tf.map_fn(lambda img: tf.image.per_image_standardization(img), x_image)
        normalized_data = sess.run(normalize, feed_dict={x_image: images})
        sess.close()
        return normalized_data

    def whiten_images(self, images):
        self.means = []
        self.stddevs = []
        for i in range(0, 3):
            mean_channel = np.mean(images[:][:][:][i])
            self.means.append(mean_channel)
            stddev_channel = np.std(images[:][:][:][i])
            self.stddevs.append(stddev_channel)
            images[:][:][:][i] = (images[:][:][:][i] - mean_channel) / stddev_channel
        return images

    def get_train_batch(self, allow_smaller_batches=False):
        return self._get_batch('train', allow_smaller_batches)

    def get_test_batch(self, allow_smaller_batches=False):
        return self._get_batch('test', allow_smaller_batches)

    def reset(self):
        self.current_idx_train = 0
        self.current_idx_test = 0
        self.permutation_train = np.random.permutation(self.num_train_items)
        self.permutation_test = np.random.permutation(self.num_test_items)

    def batch_generator(self, group, batch_size=50, limit=False, fraction=1):
        idx = 0
        data = self.train_data if group == 'train' else self.test_data
        labels = self.train_labels if group == 'train' else self.test_labels
        dataset_size = labels.shape[0]
        indices = range(dataset_size)
        np.random.shuffle(indices)
        while idx < dataset_size * fraction:
            chunk = slice(idx, idx + batch_size)
            chunk = indices[chunk]
            idx = idx + batch_size
            if not limit and idx >= dataset_size:
                np.random.shuffle(indices)
                idx = 0
            yield ([data[i] for i in chunk], [labels[i] for i in chunk])

    def _get_batch(self, data_set, allow_smaller_batches=False):
        return self.batch_generator(data_set, self.batch_size)

    def view_image_labels(self):
        from matplotlib import pyplot as plt
        seen_labels = []
        for r in range(0, len(self.train_labels)):
            label = self.train_labels[r]
            idx = np.argmax(label)
            if idx not in seen_labels:
                print(idx)
                seen_labels.append(idx)
                print(self.train_labels[r])
                plt.figure()
                plt.imshow(self.train_data[r])
                plt.show()
