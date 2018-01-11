import numpy as np
import tensorflow as tf

class GTSRB:
    WIDTH = 32
    HEIGHT = 32
    CHANNELS = 3
    OUTPUT = 43
    num_test_items = -1
    speed_limit_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    prohibitory_classes = [9, 10, 15, 16]
    derestriction_classes = [6, 32, 41, 42]
    mandatory_classes = [33, 34, 35, 36, 37, 38, 39, 40]
    danger_classes = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    unique_classes = [12, 13, 14, 17]

    def __init__(self, batch_size=128, use_augmented_data=False, normalise_data=True, whiten_data=True, seed=10):
        np.random.seed(seed)
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
        indices = np.arange(dataset_size)
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

    def count_classes(self):
        classes_counts = np.sum(self.train_data, axis=0)
        class_counts = {}
        class_counts['speed_limits'] = np.sum(classes_counts[self.speed_limit_classes])
        class_counts['prohibitory'] = np.sum(classes_counts[self.prohibitory_classes])
        class_counts['derestriction'] = np.sum(classes_counts[self.derestriction_classes])
        class_counts['mandatory'] = np.sum(classes_counts[self.mandatory_classes])
        class_counts['danger'] = np.sum(class_counts[self.danger_classes])
        class_counts['unique'] = np.sum(class_counts[self.unique_classes])
        return class_counts

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


if __name__ == '__main__':
    g = GTSRB(use_augmented_data=False, normalise_data=False, whiten_data=False)
    class_counts = g.count_classes()
    print(class_counts)