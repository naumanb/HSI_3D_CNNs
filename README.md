<h2 align="center">
Comparison of CNN Algorithms for Glioma Tumor Segmentation in Brain HS Images
</h2>
  
This project aims to compare various Convolutional Neural Network (CNN) architectures for classifying brain cancer in Hyperspectral Images (HSI). The implemented algorithms are inspired by the paper "Comparison of CNN Algorithms on Hyperspectral Image Classification in Agricultural Lands" by Tien-Heng Hsieh† and Jean-Fu Kiang*† [1]. The CNN architectures mentioned in the paper were adapted and used on the HSI Brain Cancer dataset [2]. The algorithms include 1D CNN, 2D CNN, 1D + 2D Hybrid CNN, 2D + 1D Hybrid CNN, and 3D CNN. The models are trained and evaluated using the leave-one-image-out cross-validation method on a dataset of hyperspectral brain cancer images.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.6 or higher
- Keras 2.6 or higher
- Scikit-learn 0.24 or higher
- Numpy 1.19 or higher
- Matplotlib 3.4 or higher

### Tasks

Status: Partially Complete

- [x] Loading .mat data
- [x] Preprocessing for NN input (normalizing, cropping)
- [ ] Data augmentation for unlabeled pixels
- [x] Plotting/visualizing images and labels
- [x] Leave-one-image-out implementation
- [x] Custom loss and metric functions
- [x] 1-D CNN Implementation
- [ ] 2-D CNN Implementation
- [ ] 1-D + 2-D Hybrid
- [ ] 2-D + 1-D Hybrid
- [ ] 3-D CNN Implementation
- [x] Saving results

### Installation

1. Clone the repository:
```
git clone https://github.com/naumanb/HSI_CNNs.git
cd HSI_CNNs
```

2. Install the required dependencies using pip:
```
pip install -r requirements.txt
```

## Usage

1. Place your preprocessed (denoised) hyperspectral image data in the `data` folder. The files should be in `.mat` format.

2. Execute the `main.py` script in the `src` folder to train and evaluate the CNN architectures on the dataset:

```python
python main.py
```

The script normalizes the hyperspectral image data, split it using leave-one-out cross-validation, and train each architecture on the dataset. The models' performance metrics are displayed in the terminal, and the classification reports are saved in the `results` folder.

## Results

The performance metrics for each architecture, such as precision, recall, and F1-score, are saved in a text file in the `results` folder. The metrics are compared to analyze the performance of each algorithm for the given task.

## Contributing

If you would like to contribute to this project, please fork the repository and create a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## References
[1] Hsieh, T. H., & Kiang, J. F. (2020). Comparison of CNN algorithms on hyperspectral image classification in agricultural lands. Sensors, 20(6), 1734.

[2] Fabelo, H., Ortega, S., Szolna, A., Bulters, D., Piñeiro, J. F., Kabwama, S., ... & Sarmiento, R. (2019). In-vivo hyperspectral human brain image database for brain cancer detection. IEEE Access, 7, 39098-39116.

