# Brain Tumor Detection using VGG16

## Project Overview
This project involves detecting brain tumors from MRI images using a deep learning model based on the VGG16 architecture. The model has been trained on a dataset sourced from Kaggle and employs transfer learning to achieve high accuracy.

---

## Features
- **Model Architecture**: VGG16 pre-trained on ImageNet
- **Dataset**: MRI brain images (sourced from Kaggle)
- **Preprocessing**:
  - Resizing images to 224x224
  - Normalization of pixel values
- **Transfer Learning**:
  - Fine-tuned final layers of VGG16
  - Custom classifier for binary classification (tumor vs. no tumor)
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

---

## Prerequisites
Ensure the following are installed on your system:

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Pandas
- scikit-learn
- OpenCV (for image preprocessing)

---

## Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/) and place it in the `dataset/` directory. The folder structure should look like:
https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

```
dataset/
  |- training/
  |    |- yes/   # Images with tumors
  |    |- no/    # Images without tumors
  |
  |- testing/
       |- yes/   # Images with tumors
       |- no/    # Images without tumors
```

---

## Usage

### Training the Model
1. Run the training script:
   ```bash
   python train_model.py
   ```
2. The model will be saved in the `models/` directory after training.

---

## Model Architecture
The model leverages the VGG16 architecture with the following modifications:
- Removed the top layers of VGG16.
- Added custom fully connected layers:
  - Dense(256, activation='relu')
  - Dropout(0.5)
  - Dense(1, activation='sigmoid')
---

## Visualization
- Example input images with predictions:
  - ![Example Tumor Image]
  - ![Example Non-Tumor Image]

---

## Future Work
- Enhance the dataset by including more diverse MRI images.
- Experiment with other architectures like ResNet50 or Inception.
- Develop a web-based interface for real-time predictions
---

## Contact
For queries, contact me aastha.vashishtha.2003@gmail.com
