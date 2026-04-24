# Skin Cancer Classification using Deep Learning

A deep learning project for automated melanoma detection using Convolutional Neural Networks (CNNs) and transfer learning. This project implements multiple state-of-the-art architectures to classify skin lesions as benign or malignant.

## 📋 Project Overview

Skin cancer, particularly melanoma, is one of the most common and deadly forms of cancer. Early detection significantly improves survival rates. This project leverages deep learning to assist in automated skin cancer detection from dermoscopic images.

## 🎯 Key Features

- **Multiple Model Architectures**: Implementation of CNN, ResNet50, VGG16, and ensemble methods
- **Transfer Learning**: Utilizes pre-trained models on ImageNet for better performance
- **Class Imbalance Handling**: Implements class weights to address imbalanced medical datasets
- **Data Augmentation**: Enhanced training with image augmentation techniques
- **Advanced Callbacks**: Early stopping, learning rate reduction, and model checkpointing
- **Ensemble Methods**: Combines multiple models for improved prediction accuracy

## 📊 Dataset

- **Source**: Melanoma Skin Cancer Dataset (10,000 images)
- **Platform**: Kaggle
- **Classes**: Binary classification (Benign vs Malignant)
- **Format**: RGB images
- **Split**: Training and testing sets

## 🏗️ Model Architectures

### 1. Custom CNN
- Built from scratch with multiple convolutional layers
- Batch normalization and dropout for regularization
- Optimized for skin lesion classification

### 2. ResNet50 (Transfer Learning)
- Pre-trained on ImageNet
- Fine-tuned last 50 layers for domain adaptation
- Global average pooling for feature extraction

### 3. VGG16 (Transfer Learning)
- Deep architecture with 16 layers
- Frozen base layers with trainable top layers
- Adapted for binary classification

### 4. Ensemble Models
- Combination of VGG16 + CNN + ResNet50
- Weighted averaging for final predictions
- Improved robustness and accuracy

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow / Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib / Seaborn**: Visualization
- **scikit-learn**: Metrics and utilities
- **Kaggle API**: Dataset management

## 📦 Installation

### Prerequisites
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn kaggle
```

### Dataset Setup
1. Install Kaggle API:
```bash
pip install kaggle
```

2. Upload your `kaggle.json` credentials file

3. Download the dataset:
```bash
kaggle datasets download -d <dataset-name>
unzip <dataset-name>.zip
```

## 🚀 Usage

### Training Models

#### Basic CNN
```python
cnn_model = build_cnn(input_shape, num_classes)
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(train_ds, validation_data=test_ds, epochs=epochs, class_weight=class_weights)
```

#### ResNet50
```python
base_model = ResNet50(weights="imagenet", include_top=False)
# Fine-tune last 50 layers
for layer in base_model.layers[-50:]:
    layer.trainable = True
```

#### VGG16
```python
vgg_base = VGG16(weights="imagenet", include_top=False)
model = build_vgg_model(vgg_base, num_classes)
```

### Notebooks
- `skin_cancer_1.ipynb`: Initial experiments with CNN, ResNet50, VGG16, and ensemble methods
- `skin_cancer_2.ipynb`: Advanced implementations and model variations

## 📈 Model Training Features

### Callbacks Implemented
- **EarlyStopping**: Prevents overfitting by monitoring validation loss
- **ReduceLROnPlateau**: Dynamically adjusts learning rate during training
- **ModelCheckpoint**: Saves best model based on validation performance

### Data Preprocessing
- Image normalization (0-1 scaling)
- Uniform image resizing
- Data augmentation (rotation, flip, zoom)
- Class weight balancing

## 🎓 Model Evaluation

Models are evaluated using:
- Accuracy metrics
- Confusion matrices
- Classification reports
- Loss and accuracy curves

## 📝 Project Structure

```
├── skin_cancer_1.ipynb       # Main training notebook
├── skin_cancer_2.ipynb       # Advanced experiments
├── data/
│   └── melanoma_cancer_dataset/
│       ├── train/
│       └── test/
├── models/
│   └── best_model.keras      # Saved best model
└── README.md
```

## 🔬 Research & Applications

This project demonstrates:
- Application of transfer learning in medical imaging
- Handling imbalanced medical datasets
- Multi-model ensemble approaches
- Real-world deployment considerations for clinical use

## ⚠️ Disclaimer

This project is for educational and research purposes only. It is not intended to replace professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset providers on Kaggle
- TensorFlow and Keras teams
- Medical imaging research community
- Pre-trained model developers (ImageNet)

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Note**: Ensure you have appropriate computational resources (GPU recommended) for training deep learning models. Google Colab with GPU runtime is recommended for running the notebooks.
