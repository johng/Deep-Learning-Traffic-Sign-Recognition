import numpy as np
from matplotlib import pyplot as plt


class GTSRB:
    WIDTH = 32
    HEIGHT = 32
    CHANNELS = 3
    OUTPUT = 43
    num_test_items = -1

    def __init__(self, batch_size=128, use_augmented_data=False):
        dataset = np.load('gtsrb_dataset.npz')

        self.train_data = dataset['X_train']
        self.train_labels = dataset['y_train']
        print('Original data:   {}'.format(self.train_data.shape))
        print('Original labels: {}'.format(self.train_labels.shape))

        if use_augmented_data:
            augmented_dataset = np.load('extended_dataset.npz')
            self.train_data.append(augmented_dataset['images'])
            self.train_labels.append(augmented_dataset['labels'])
            print('All data:   {}'.format(self.train_data.shape))
            print('All labels: {}'.format(self.train_labels.shape))

        self.test_data = dataset['X_test']
        self.test_labels = dataset['y_test']

        self.num_train_items = len(self.train_labels)
        self.num_test_items = len(self.test_labels)

        self.batch_size = batch_size

        self.permutation_train = np.random.permutation(self.num_train_items)
        self.permutation_test = np.random.permutation(self.num_test_items)

        self.current_idx_test = 0
        self.current_idx_train = 0

    def whiten_images(self, images):
        # sess = tf.Session()
        # x_image = tf.placeholder(tf.float32, [None, gtsrb.WIDTH, gtsrb.HEIGHT, gtsrb.CHANNELS])
        images = np.mean(images, axis=0)
        cov = np.dot(images.T, images) / images.shape[0]
        U, S, V = np.linalg.svd(cov)
        images_rot = np.dot(images, U)
        whitened = images_rot / np.sqrt(S + 1e-5)
        # whitened_images = sess.run(augmented_data, feed_dict={x_image: images})
        # sess.close()
        return whitened

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
