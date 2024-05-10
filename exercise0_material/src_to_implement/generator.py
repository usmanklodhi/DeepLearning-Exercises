import os.path
import json
import scipy.misc
import random
import numpy as np
import matplotlib.pyplot as plt


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.mirror_axis=None

        assert (isinstance(file_path, str)), "The file_path must be a string."
        self.file_path = file_path
        assert (isinstance(label_path, str)), "The label_path must be a string."
        self.label_path = label_path
        assert (isinstance(batch_size, int)), "The batch_size must be an integer."
        self.batch_size = batch_size
        assert isinstance(image_size, list) and all(isinstance(item, int) for item in image_size), ("The image_size "
                                                                                                    "must be a list "
                                                                                                    "of integers.")
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.class_dict = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }

        self.current_epoch = 0
        self.current_batch = 0


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method
        pass
        #return images, labels

    def augment(self, img):

        # Check if rotation augmentation is enabled
        if self.rotation:
            # Choose a random angle from the allowed rotation angles
            angle = random.choice([90, 180, 270])
            # Rotate the image by the selected angle and Divide angle by 90 to get number of rotations
            img = np.rot90(img, angle // 90, (0, 1))

            # Check if mirroring augmentation is enabled
        if self.mirroring:
            # Flip the image along a random axis selected from self.mirror_axis
            img = np.flip(img, random.choice(self.mirror_axis))

        return img

    def current_epoch(self):
        return self.current_epoch

    def class_name(self, x):
        return self.class_dict[x]

    def show(self):
        images, labels = self.next()
        subplot_count = np.ceil(np.sqrt(self.batch_size))
        for i, (image, label) in enumerate(zip(images, labels)):
            # Subplot indices start at 1, not 0
            plt.subplot(subplot_count, subplot_count, i + 1)
            plt.title(self.class_name(label))
            plt.show(image)
        plt.show()


