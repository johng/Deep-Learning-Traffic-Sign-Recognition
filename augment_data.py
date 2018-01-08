from GTSRB import GTSRB
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import sys

ia.seed(1)

augmentation_sequence = iaa.SomeOf(1, [
    iaa.CropAndPad(
        px=((0, 10), (0, 10), (0, 10), (0, 10)),
        pad_mode=ia.ALL,
        pad_cval=(0, 128)
    ),
    iaa.Dropout((0.0, 0.05)),
    # iaa.WithColorspace(from_colorspace='RGB', to_colorspace='HSV', children=iaa.WithChannels(2, iaa.Add((0,10)))),
    iaa.Add((-50, 50)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
    iaa.AverageBlur(k=((4, 8), (1, 3))),
    iaa.PerspectiveTransform(scale=(0.01, 0.2)),
    iaa.Affine(rotate=(-30, 30), scale=(0.75, 1.25))
], random_order=True)


def view_augmented_image(images, idx):
    # set SCIPY_PIL_IMAGE_VIEWER env variable to an image viewer executable
    augmentation_sequence.show_grid((images * 255.0)[idx], rows=8, cols=8)


def generate_extended_set(gtsrb):
    print('Original images: {}'.format(gtsrb.train_data.shape))
    print('Original labels: {}'.format(gtsrb.train_labels.shape))
    augmented_images, augmented_labels = augmentation_sequence.augment_images(
        gtsrb.train_data * 255.0) / 255.0, gtsrb.train_labels
    print('Augmented images: {}'.format(augmented_images.shape))
    print('Augmented labels: {}'.format(augmented_labels.shape))
    np.savez('extended_dataset', images=augmented_images, labels=augmented_labels)


if __name__ == '__main__':
    data = GTSRB(use_augmented_data=False)
    if sys.argv[1] == 'generate':
        print('Generating augmented data')
        generate_extended_set(data)
    elif sys.argv[1] == 'show':
        print('Showing augmented image {}'.format(sys.argv[2]))
        try:
            idx = int(sys.argv[2])
            view_augmented_image(data.train_data, idx)
        except ValueError:
            print('Invalid image index {}'.format(sys.argv[2]))
