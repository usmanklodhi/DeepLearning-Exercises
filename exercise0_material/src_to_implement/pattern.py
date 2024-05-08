import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        assert resolution % (2 * tile_size) == 0, "Resolution must be evenly dividable by 2 * tile_size."
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        # Create an empty numpy array initialized with zeros
        # dtype=np.uint8 is an unsigned 8-bit integer, which means it can store values ranging from 0 to 255
        self.output = np.zeros((self.resolution, self.resolution), dtype=np.uint8)

        # Creating a checkerboard pattern by filling the array
        for i in range(0, self.resolution, self.tile_size):
            for j in range(0, self.resolution, self.tile_size):
                # Determine colour of the tile based on its position
                # black (0) if both indices are even or both are odd, and vice versa
                if (i // self.tile_size) % 2 == (j // self.tile_size) % 2:
                    self.output[i:i + self.tile_size, j:j + self.tile_size] = 0
                else:
                    self.output[i:i + self.tile_size, j:j + self.tile_size] = 1

        # Caller receives copy of the array rather than a reference to the original array stored within the object
        return self.output.copy()

    def show(self):
        if self.output is None:
            print("The pattern does not exist.")
        else:
            plt.imshow(self.output, cmap='gray')
            plt.axis('off')
            plt.show()


class Spectrum:
    def __init__(self, resolution):
        assert isinstance(resolution, int), "The resolution must be an integer."
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros((self.resolution, self.resolution, 3), dtype=np.float64)
        # Use np.linspace which creates a linearly spaced array
        # Between start, end value over a certain number of points
        self.output[:, :, 0] = np.linspace(0, 1, self.resolution)  # Red channel (creates a gradient from non-red to
        # red)
        self.output[:, :, 2] = np.linspace(1, 0, self.resolution)  # Blue channel (creates a reverse gradient from
        # blue to non-blue)
        self.output[:, :, 1] = np.linspace(0, 1, self.resolution).reshape(self.resolution, 1)  # Green channel (
        # creates a vertical gradient from non-green to green)

        return self.output.copy()

    def show(self):
        if self.output is None:
            print("The pattern does not exist.")
        else:
            plt.imshow(self.output)
            plt.axis('off')
            plt.show()