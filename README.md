# Image Quality Assessment (IQA) Framework User Guide

This guide provides an overview of the process for setting up and using the Image Quality Assessment (IQA) framework for predicting image quality based on various datasets. The framework leverages deep learning models, including ResNet50, U-2-Net, and multi-modal feature fusion techniques.

## 1. Overview

The framework focuses on the assessment of image quality in real-world conditions, particularly for User-Generated Content (UGC) images. It integrates saliency-guided attention mechanisms and feature fusion using CNNs and Transformers to provide a comprehensive and perceptually aligned quality score.

Key components:
- **IQA_dataloader**: A custom dataset loader to handle multiple IQA datasets like **CLIVE**, **SPAQ**, and **KonIQ-10K**.
- **ResEVIQA Model**: A multi-modal feature fusion network that combines features from CNNs (ResNet50) and Transformers, enhanced with saliency-guided attention mechanisms.
- **U-2-Net**: A model for saliency detection, used to crop images based on salient regions.

## 2. Datasets

The following datasets are supported for training and testing:

- **CLIVE Dataset**:
  - **Description**: Contains real-world images for quality assessment.
  - **Dataset URL**: [CLIVE Dataset](https://database.mmsp-kn.de/).

- **SPAQ Dataset**:
  - **Description**: Includes images with various distortion types and quality levels.
  - **Dataset URL**: [SPAQ Dataset](https://www4.cs.unc.edu/~rsw/SPAQ/).

- **KonIQ-10K Dataset**:
  - **Description**: Contains images used for subjective quality evaluation with MOS (Mean Opinion Score) annotations.
  - **Dataset URL**: [KonIQ-10K Dataset](https://database.mmsp-kn.de/).

### Data Preparation:
- Download the datasets from the provided URLs.
- Ensure the images are stored in a directory that the script can access.
- Use the provided CSV files containing the Mean Opinion Scores (MOS) for the corresponding images.

## 3. Installation

### Dependencies
Before running the framework, ensure the following libraries are installed:
- `torch` (PyTorch)
- `torchvision`
- `PIL`
- `numpy`
- `pandas`
- `opencv-python`
- `tqdm`

To install the required dependencies:
```bash
pip install torch torchvision pandas numpy opencv-python tqdm
```

### U-2-Net Model
The U-2-Net model is used for saliency detection. You need to download the pre-trained U-2-Net weights from [U-2-Net repository](https://github.com/xuebinqin/U-2-Net) and place them in the appropriate directory.

```bash
git clone https://github.com/xuebinqin/U-2-Net.git
cd U-2-Net
wget https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth
```

Ensure the `u2net.pth` file is placed in the directory specified in your code.

## 4. Running the Framework

### Step 1: Prepare the Data

1. **Load the datasets**:
   - For each dataset, you need to specify the dataset directory and the corresponding CSV file that contains the MOS annotations.

   Example:
   ```python
   data_dir = "path_to_dataset_directory"
   csv_path = "path_to_csv_file.csv"
   ```

2. **Use the `IQA_dataloader` to load images**:
   - The `IQA_dataloader` class handles different datasets (e.g., CLIVE, SPAQ, KonIQ-10K).
   - Example initialization:
   ```python
   from IQADataset import IQA_dataloader
   dataset = IQA_dataloader(data_dir, csv_path, database='Koniq10k')
   ```

### Step 2: Preprocessing the Images

The images in the datasets need to be preprocessed. This includes resizing, normalization, and applying any augmentations as specified in the `transform` parameter.

- **Saliency Detection with U-2-Net**: Use the `DatabaseCROPⅢ.py` script to generate saliency maps and crop images based on salient regions.
- **Process Images**:
   ```python
from DatabaseCROPⅢ import process_dataset, U2NET
   model = U2NET(in_ch=3, out_ch=1).to(device)
   model.load_state_dict(torch.load('path_to_u2net.pth', map_location=device))  # Load pre-trained U-2-Net model
   process_dataset("input_data_directory", "output_data_directory", model)
   ```

### Step 3: Model Training

1. **Model Initialization**:
   The model uses **ResEVIQA**, which is a multi-modal network combining CNN and Transformer features.
   ```python
   from ResEV import ResEVIQA
   model = ResEVIQA().to(device)
   ```

2. **Training the Model**:
   - Ensure that your dataset is loaded properly with the `IQA_dataloader`.
   - Train the model on your dataset by defining the loss function (e.g., MSE loss for quality prediction) and optimizer (e.g., Adam).

### Step 4: Evaluate the Model

Once trained, you can evaluate the model on test datasets to predict the image quality scores. The framework provides functions for calculating and displaying quality scores for images based on the trained model.

```python
# Example evaluation
input_image = torch.randn(2, 3, 224, 224).to(device)  # Random input
output_quality_score = model(input_image)
print(f"Predicted Quality Score: {output_quality_score}")
```

## 5. Example Usage

```python
from IQADataset import IQA_dataloader
from ResEV import ResEVIQA
from DatabaseCROPⅢ import process_dataset, U2NET

# Initialize dataset loader
dataset = IQA_dataloader("path_to_data", "path_to_csv", database="Koniq10k")

# Load pre-trained U-2-Net model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = U2NET(in_ch=3, out_ch=1).to(device)
model.load_state_dict(torch.load('path_to_u2net.pth', map_location=device))

# Process and crop images
process_dataset("input_data_directory", "output_data_directory", model)

# Initialize ResEVIQA model
iqa_model = ResEVIQA().to(device)

# Predict quality scores
image = torch.randn(1, 3, 224, 224).to(device)
predicted_quality = iqa_model(image)
print(f"Predicted Image Quality: {predicted_quality}")
```

## 6. Conclusion

This guide provides instructions on how to set up and use the Image Quality Assessment framework, incorporating multiple datasets and advanced deep learning models. With saliency-guided attention mechanisms and multi-modal feature fusion, the framework is capable of providing perceptually aligned image quality predictions.
