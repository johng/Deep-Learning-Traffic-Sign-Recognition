import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt

ia.seed(1)


class gtsrb:
    WIDTH = 32
    HEIGHT = 32
    CHANNELS = 3
    OUTPUT = 43
    nTestSamples = 200

    def __init__(self, batch_size=128, use_extended=False, generate_extended=False):

        dataset = np.load('gtsrb_dataset.npz')
        if generate_extended:
            self.trainData = dataset['X_{0:s}'.format('train')]
            self.trainLabels = dataset['y_{0:s}'.format('train')]
            self.generate_extended_set()
        if use_extended:
            extended = np.load('extended_dataset.npz')
            self.trainData = extended['arr_0']
            self.trainLabels = extended['arr_1']
            # assert np.all(t_a, t_b)
            print("Extended dataset {}".format(self.trainData.shape))
            print("Extended labels {}".format(self.trainLabels.shape))
        else:
            self.trainData = dataset['X_{0:s}'.format('train')]
            self.trainLabels = dataset['y_{0:s}'.format('train')]
        self.testData = dataset['X_{0:s}'.format('test')]
        self.testLabels = dataset['y_{0:s}'.format('test')]

        self.nTrainSamples = len(self.trainLabels)
        self.nTestSamples = len(self.testLabels)

        self.batchSize = batch_size

        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)

        self.currentIndexTest = 0
        self.currentIndexTrain = 0

    def augment_images(self, images, classes):
        seq = iaa.SomeOf(2, [
            iaa.CropAndPad(
                px=((0, 10), (0, 10), (0, 10), (0, 10)),
                pad_mode=ia.ALL,
                pad_cval=(0,128)
            ),
            # iaa.Sequential([
            #     iaa.ChangeColorspace(from_colorspace='RGB', to_colorspace='YCrCb'),
            #     iaa.WithChannels(0, iaa.Add((-30, 30))),
            #     iaa.ChangeColorspace(from_colorspace='YCrCb', to_colorspace='RGB')
            # ]),
            iaa.AdditiveGaussianNoise(scale=(0, 0.03*255)),
            iaa.AverageBlur(k=((4,8), (1,3))),
            iaa.PerspectiveTransform(scale=(0.01, 0.2))
        ], random_order=True)

        augmented_images = seq.augment_images(images*255.0)

        # set SCIPY_PIL_IMAGE_VIEWER env variable to an image viewer executable
        #seq.show_grid((images*255)[2000], rows=8, cols=8)

        # fig = plt.figure()
        # a = fig.add_subplot(2, 2, 1)
        # a.imshow(augmented_images[0], interpolation='nearest')
        # b = fig.add_subplot(2, 2, 2)
        # b.imshow(augmented_images[1], interpolation='nearest')
        # b = fig.add_subplot(2, 2, 3)
        # b.imshow(augmented_images[2], interpolation='nearest')
        # b = fig.add_subplot(2, 2, 4)
        # b.imshow(augmented_images[3], interpolation='nearest')
        # plt.show()

        return np.concatenate((images, augmented_images/255.0)), np.concatenate((classes, classes))

    # def augment_images(self, images, classes):
    #     sess = tf.Session()
    #     x_image = tf.placeholder(tf.float32, [None, gtsrb.WIDTH, gtsrb.HEIGHT, gtsrb.CHANNELS])
    #     rotate_images = tf.map_fn(lambda x: tf.contrib.image.rotate(x, tf.random_uniform([], -0.26, 0.26)), x_image)
    #     translate_images = tf.map_fn(lambda x: tf.contrib.image.transform(x, [1, 0,
    #                                                                           tf.random_uniform([], -2, 2), 0, 1,
    #                                                                           tf.random_uniform([], -2, 2), 0, 0]),
    #                                  x_image)
    #
    #     augmented_data = tf.concat([x_image, rotate_images, translate_images], axis=0)
    #     extended_data = sess.run(augmented_data, feed_dict={x_image: images})
    #     sess.close()
    #     return extended_data, np.concatenate((classes, classes, classes))

    def generate_extended_set(self):
        h_flip_invariant_classes = [17, 12, 13, 15, 35]
        v_flip_invariant_classes = []
        new_trainData = []
        new_trainLabels = []

        for idx, img in enumerate(self.trainData):
            label = np.argmax(self.trainLabels[idx])
            if label in h_flip_invariant_classes:
                flipped = np.fliplr(img)
                new_trainData.append(flipped)
                new_trainLabels.append(self.trainLabels[idx])
            if label in v_flip_invariant_classes:
                flipped = np.flip(img, 1)
                new_trainData.append(flipped)
                new_trainLabels.append(self.trainLabels[idx])

        # extended_trainData = np.concatenate((self.trainData, np.array(new_trainData)), axis=0)
        # extended_trainLabels = np.concatenate((self.trainLabels, np.array(new_trainLabels)), axis=0)
        augmented_images, augmented_labels = self.augment_images(self.trainData, self.trainLabels)
        print("Extended dataset from {} to {}".format(self.trainData.shape, augmented_images.shape))
        print("Extended labels from {} to {}".format(self.trainLabels.shape, augmented_labels.shape))
        np.savez('extended_dataset', augmented_images, augmented_labels)

    def get_train_batch(self, allow_smaller_batches=False):
        return self._get_batch('train', allow_smaller_batches)

    def get_test_batch(self, allow_smaller_batches=False):
        return self._get_batch('test', allow_smaller_batches)

    def reset(self):

        self.currentIndexTrain = 0
        self.currentIndexTest = 0
        self.pTrain = np.random.permutation(self.nTrainSamples)
        self.pTest = np.random.permutation(self.nTestSamples)

    def batch_generator(self, group, batch_size=50, limit=False, fraction=1):

        idx = 0
        data = self.trainData if group == 'train' else self.testData
        labels = self.trainLabels if group == 'train' else self.testLabels

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
        return self.batch_generator(data_set, self.batchSize)

    def view_image_labels(self):
        seen_labels = []
        for r in range(0, len(self.trainLabels)):
            label = self.trainLabels[r]
            idx = np.argmax(label)
            if idx not in seen_labels:
                print(idx)
                seen_labels.append(idx)
                print(self.trainLabels[r])
                plt.figure()
                plt.imshow(self.trainData[r])
                plt.show()


if __name__ == '__main__':
    data = gtsrb()
    data.augment_images_2(data.trainData, data.trainLabels)
