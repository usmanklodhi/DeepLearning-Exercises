import os.path
import json
import scipy.misc
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # self.mirror_axis=None

        if mirroring:
            self.mirror_axis = [0, 1]  # Set this to valid axis indices for mirroring

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

        self.current_epoch_no = 0
        self.current_batch = 0

        label_json = open(self.label_path, "rb")  # read, binary
        self.labels = json.load(label_json)  # convert json data into python object
        label_json.close()

        image_files = os.listdir(self.file_path)  # list all entries in a specified directory
        self.image_names = [img_file[:-4] for img_file in image_files]  # remove file extension

        if self.shuffle:
            random.shuffle(self.image_names)


    def next(self):
        # Calculate the starting index of the current batch
        current_image_no = self.current_batch * self.batch_size

        # Determine the end index for the current batch
        last_img_num = current_image_no + self.batch_size
        if last_img_num > len(self.image_names):
            # Handle the end of dataset and optional reshuffling
            if self.shuffle:
                random.shuffle(self.image_names)
            last_img_num = len(self.image_names)
            self.current_epoch_no += 1
            self.current_batch = 0

        # Select batch names
        image_batch_names = self.image_names[current_image_no:last_img_num]

        # If the batch is smaller than batch_size, add images from the front
        if len(image_batch_names) < self.batch_size and last_img_num == len(self.image_names):
            num_of_less_images = self.batch_size - len(image_batch_names)
            image_batch_names += self.image_names[:num_of_less_images]

        # Load and process images and labels
        images = []
        labels = []
        for image_name in image_batch_names:
            image = np.load(self.file_path + image_name + ".npy")
            image = self.augment(image)
            image = cv2.resize(image, tuple(self.image_size[:-1]))
            label = self.labels[image_name]
            images.append(image)
            labels.append(label)

        # Convert lists to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        # Increment the batch number for the next call
        self.current_batch += 1

        return images, labels

    def augment(self, img):
        if self.rotation:
            # Choose a random angle from the allowed rotation angles
            angle = random.choice([90, 180, 270])
            # Rotate the image by the selected angle and Divide angle by 90 to get number of rotations
            img = np.rot90(img, angle // 90, (0, 1))
        if self.mirroring:
            # Flip the image along a random axis selected from self.mirror_axis
            img = np.flip(img, random.choice(self.mirror_axis))

        return img

    def current_epoch(self):
        return self.current_epoch_no

    def class_name(self, x):
        return self.class_dict[x]

    def show(self):
        images, labels = self.next()
        batch_sqrt = int(np.ceil(np.sqrt(self.batch_size)))
        counter = 0
        for image, label in zip(images, labels):
            counter += 1
            plt.subplot(batch_sqrt, batch_sqrt, counter)
            plt.title(self.class_name(label))
            plt.axis('off')
            plt.imshow(image)
        plt.show()


if __name__ == "__main__":
    gen = ImageGenerator("exercise_data/", "Labels.json", 60, [32, 32, 3], rotation=False, mirroring=False, shuffle=False)
    gen.show()