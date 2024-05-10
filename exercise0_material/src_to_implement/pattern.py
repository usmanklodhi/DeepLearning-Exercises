import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        assert resolution % (2 * tile_size) == 0, "Resolution must be evenly dividable by 2 * tile_size."
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        # Create column and row index arrays for the entire image.
        # These arrays will represent the pixel coordinates in the checkerboard.
        grid_x = np.arange(self.resolution)[:, None]  # Column vector of indices (0 to resolution-1)
        grid_y = np.arange(self.resolution)[None, :]  # Row vector of indices (0 to resolution-1)

        # Convert pixel indices to tile indices.
        # By integer division by `tile_size`, indices are transformed into tile coordinates,
        # where each tile contains `tile_size` pixels.
        grid_x = grid_x // self.tile_size  # Each value tells which tile row a pixel belongs to
        grid_y = grid_y // self.tile_size  # Each value tells which tile column a pixel belongs to

        # Determine the color of each tile based on its position.
        # If both the row and column indices of a tile have the same parity (both even or both odd),
        # the tile is black (0). If they differ, the tile is white (1).
        self.output = np.where((grid_x % 2) == (grid_y % 2), 0, 1)  # Create the checkerboard pattern

        # np.where: used for selecting elements from two arrays based on the condition
        # If parity is same, insert black tile and vice versa

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


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position

    def draw(self):
        # Generate arrays of x and y coordinates from 0 to resolution - 1
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        # Create a meshgrid of x and y coordinates
        x, y = np.meshgrid(x, y)

        # Calculate distances from each point to the center of the circle, distances = np.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)
        distances = np.sqrt((x - self.position[0]) ** 2 + (y - self.position[1]) ** 2)

        # Create an output array where points inside the circle have value 1, and outside have value 0
        self.output = np.where(distances <= self.radius, 1, 0)

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='binary')
        plt.axis('off')
        plt.show()
