import os
import scipy.io
import numpy as np

# Load the hyperspectral image data from the .mat files
def load_all_mat_data(data_folder, variable_name='av', label_variable_name='gtMap'):
    mat_files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]
    hyperspectral_images = []
    ground_truth_labels = []

    for mat_file in mat_files:
        mat_file_path = os.path.join(data_folder, mat_file)
        mat_data = scipy.io.loadmat(mat_file_path)

        # Get the hyperspectral image data and the ground truth labels
        hyperspectral_image_data = mat_data[variable_name]
        hyperspectral_images.append(hyperspectral_image_data)

        ground_truth_label_data = mat_data[label_variable_name]
        ground_truth_labels.append(ground_truth_label_data)

    return np.array(hyperspectral_images, dtype=object), np.array(ground_truth_labels, dtype=object)

# Preprocess the hyperspectral image data
def preprocess_data(hyperspectral_image_data, label_data):
    preprocessed_data = []
    filtered_label_data = []
    image_indices = []

    for i, (hs_image, gt_labels) in enumerate(zip(hyperspectral_image_data, label_data)):
        if np.any(gt_labels == 2):  # Check if the image contains any pixels with a label value of 2 (tumor)

            # Normalize the pixel values between 0 and 1
            normalized_hs_image = hs_image / np.max(hs_image)

            preprocessed_data.append(normalized_hs_image)
            filtered_label_data.append(gt_labels)
            image_indices.append(i)
    
    return np.array(preprocessed_data, dtype=object), np.array(filtered_label_data, dtype=object), np.array(image_indices)

# Determine the minimum dimension of the images
def find_min_dimension(images):
    min_dim = float('inf')
    for img in images:
        height, width = img.shape[:2]
        min_dim = min(min_dim, height, width)
    return min_dim

# Crop the center square of the image
def crop_center_square(image, target_dim):
    height, width = image.shape[:2]
    start_row = (height - target_dim) // 2
    start_col = (width - target_dim) // 2
    cropped_image = image[start_row:start_row + target_dim, start_col:start_col + target_dim]
    return cropped_image

# Crop all images to a square
def crop_all_images(images):
    min_dim = find_min_dimension(images)
    cropped_images = [crop_center_square(img, min_dim) for img in images]
    return cropped_images, min_dim