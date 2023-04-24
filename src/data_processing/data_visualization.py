import matplotlib.pyplot as plt
from data_processing.data_preprocessing import crop_center_square
import numpy as np

# Plot hyperspectral image
def visualize_rgb_image(hyperspectral_image, r_band, g_band, b_band):
    rgb_image = hyperspectral_image[:, :, [r_band, g_band, b_band]]
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())  # Normalize the image
    plt.imshow(rgb_image)
    plt.show()

# Plot jpg image from path
def display_jpg_image(path, target_dim):
    img = plt.imread(path)
    cropped_img = crop_center_square(img, target_dim)
    plt.imshow(cropped_img)
    plt.show()

# Plot ground truth labels
def visualize_ground_truth(labels, target_dim):
    color_map = np.zeros((*labels.shape, 3))
    color_map[labels == 0] = [1, 1, 1]  # Unlabeled pixels are white
    color_map[labels == 1] = [0, 1, 0]  # Green for Normal
    color_map[labels == 2] = [1, 0, 0]  # Red for Tumor
    color_map[labels == 3] = [0, 0, 1]  # Blue for Hypervascular
    color_map[labels == 4] = [0, 0, 0]  # Black for Background tissue

    cropped_map = crop_center_square(color_map, target_dim)
    
    plt.imshow(cropped_map)
    plt.show()