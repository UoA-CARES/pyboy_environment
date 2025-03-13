
import numpy as np

class ImageStorage:
    def __init__(self):
        self.images = []

    def add_image(self, new_image):
        if not isinstance(new_image, np.ndarray) or new_image.ndim != 2:
            raise ValueError("Image must be a 2D numpy array.")
        
        # Calculate uniqueness by finding the max difference
        uniqueness = self.calculate_uniqueness(new_image)
        
        # Add the new image to the list
        self.images.append(new_image)
        
        return uniqueness

    def calculate_uniqueness(self, new_image):
        if not self.images:
            return 256.0  # No images means it's completely unique

        min_diff = 256.0
        for img in self.images:
            # Calculate the difference (using absolute difference)
            diff = np.abs(new_image - img)
            # Get the maximum difference for this pair
            min_diff = min(min_diff, np.mean(diff))
        
        return min_diff