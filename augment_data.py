from GTSRB import GTSRB
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import sys
try:
    from future_builtins import zip
except ImportError:
    pass

ia.seed(1)

augmentations = iaa.SomeOf(1, [
    iaa.CropAndPad(
        px=((0, 10), (0, 10), (0, 10), (0, 10)),
        pad_mode=ia.ALL,
        pad_cval=(0, 128)
    ),
    #iaa.CoarseDropout(p=(0.05, 0.2), size_percent=(0.15, 0.20)),
    # iaa.WithColorspace(from_colorspace='RGB', to_colorspace='HSV', children=iaa.WithChannels(2, iaa.Add((0,10)))),
    iaa.Add((-50, 50)),
    #iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
    iaa.AverageBlur(k=((4, 8), (1, 3))),
    iaa.PerspectiveTransform(scale=(0.01, 0.2)),
    iaa.Affine(rotate=(-30, 30), scale=(0.75, 1.25))
], random_order=True)

flipseq = iaa.Fliplr(1.0)

def view_augmented_image(images, idx):
    # set SCIPY_PIL_IMAGE_VIEWER env variable to an image viewer executable
    augmentations.show_grid((images * 255.0)[idx], rows=8, cols=8)


def flip_invariant_images(images, labels):
    h_flip_invariant_classes = [17, 12, 13, 15, 35]

    flipped_images = []
    flipped_labels = []

    for idx, img in enumerate(images):
        label = np.argmax(labels[idx])
        if label in h_flip_invariant_classes:
            flipped_images.append(flipseq.augment_image(img * 255.0) / 255.0)
            flipped_labels.append(labels[idx])

    return np.stack(flipped_images, axis=0), np.stack(flipped_labels, axis=0)


def generate_extended_set(gtsrb):
    print('Original images: {}'.format(gtsrb.train_data.shape))
    print('Original labels: {}'.format(gtsrb.train_labels.shape))
    flipped_images, flipped_labels = flip_invariant_images(gtsrb.train_data, gtsrb.train_labels)
    augmented_images, augmented_labels = augmentations.augment_images(gtsrb.train_data * 255.0) / 255.0, \
                                         gtsrb.train_labels
    augmented_images = np.append(augmented_images, flipped_images, axis=0)
    augmented_labels = np.append(augmented_labels, flipped_labels, axis=0)
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
