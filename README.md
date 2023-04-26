<h2 align="center">
Comparison of CNN Algorithms for Tumor Segmentation in HS Images
</h2>
  
This project aims to compare various Convolutional Neural Network (CNN) architectures for classifying brain cancer in Hyperspectral Images (HSI). The implemented architectures include 1D CNN, 2D CNN, 1D + 2D Hybrid CNN, 2D + 1D Hybrid CNN, and 3D CNN. The models are trained and evaluated using the leave-one-out cross-validation method on a dataset of hyperspectral brain cancer images.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.6 or higher
- Keras 2.6 or higher
- Scikit-learn 0.24 or higher
- Numpy 1.19 or higher
- Matplotlib 3.4 or higher

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

The script will normalize the hyperspectral image data, split it using leave-one-out cross-validation, and train each architecture on the dataset. The models' performance metrics will be displayed in the terminal, and the classification reports will be saved in the `results` folder.

## Results

The performance metrics for each architecture, such as precision, recall, and F1-score, will be saved in a text file in the `results` folder. You can compare these metrics to analyze the performance of each architecture for the given task.

## Contributing

If you would like to contribute to this project, please fork the repository and create a pull request with your proposed changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.


