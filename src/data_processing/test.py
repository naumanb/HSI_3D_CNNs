import os
import scipy.io
import numpy as np

data_folder = 'data'
mat_files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]
hyperspectral_images = []

mat_file_path = os.path.join(data_folder, '008-02.mat')

# for mat_file in mat_files:
#     mat_file_path = os.path.join(data_folder, mat_file)
#     mat_data = scipy.io.loadmat(mat_file_path)
#     hyperspectral_image_data = mat_data[variable_name]
#     hyperspectral_images.append(hyperspectral_image_data)