import numpy as np
import os, sys
from data_processing.data_preprocessing import load_all_mat_data, preprocess_data, crop_all_images
from data_processing.data_splitting import get_leave_one_out_splits
from data_processing.data_visualization import visualize_rgb_image, visualize_ground_truth, display_jpg_image
from evaluate import train_and_evaluate

from models.cnn_models import build_2d_cnn, build_3d_cnn, build_unet, build_resnet50
from evaluate import train_and_evaluate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.save_report import save_outputs

# Defining path to data folder
data_folder = 'data'

# Get the list of .mat files in the data folder
mat_files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]

# Load and preprocess the hyperspectral image data
# Note: Only images with tumor pixels are used (12 images)
image_data, label_data = load_all_mat_data(data_folder)
preprocessed_data, filtered_labels, image_indices = preprocess_data(image_data, label_data)

# Crop the images to a square
cropped_images, min_dim = crop_all_images(preprocessed_data)
cropped_label_data, min_label_dim = crop_all_images(filtered_labels)

# Raise an error if min_dim is not equal to min_label_dim
if min_dim != min_label_dim:
    raise ValueError('Image dimension and label dimension do not match.')

## Visualize a sample of the preprocessed data and labels (Inactive)
# sample_image_index = 0
# red_band = 74  # Index of the red band
# green_band = 27  # Index of the green band
# blue_band = 11  # Index of the blue band
# visualize_rgb_image(preprocessed_data[sample_image_index], red_band, green_band, blue_band)

# OR

# Visualize the image using stored jpg data
sample_image_index = image_indices[0]
image_dir = 'images'
sample_image_path = os.path.join(image_dir, mat_files[sample_image_index].replace('.mat', '.jpg'))
# display_jpg_image(sample_image_path, min_dim)

## Visualize the ground truth map for the sample image
# visualize_ground_truth(filtered_labels[sample_image_index], min_label_dim)

# Initialize a list to store the predictions from each fold
all_predictions = []

architectures = ['U-Net']
num_channels = cropped_images[0].shape[2]
input_shape = (324, 324, 128) # Manually set the input shape for variable batch sizing
num_classes = 4 
epochs = 50
batch_size = 16

# Train and evaluate the model for each architecture
for arch in architectures:
    print(f"Training and evaluating {arch}...")

    if arch == '2D_CNN':
        model_builder = build_2d_cnn
    elif arch == '3D_CNN':
        model_builder = build_3d_cnn
    elif arch == 'U-Net':
        model_builder = build_unet
    elif arch == 'ResNet50':
        model_builder = build_resnet50


    avg_metrics, report, y_true, y_pred = train_and_evaluate(model_builder, input_shape, num_classes, cropped_images, cropped_label_data, epochs, batch_size)
    print(avg_metrics)
    print(report)

    # Saving the classification report to results folder
    output_folder = 'results'
    file_name = 'classification_report.txt'
    save_outputs(y_true, y_pred, output_folder, file_name, arch)
