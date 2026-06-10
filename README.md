# SAGF-IQA: Saliency-Guided Adaptive Global-Local Framework for Blind Image Quality Assessment

This repository contains the official PyTorch implementation of the paper:  
**"SAGF-IQA: Saliency-Guided Adaptive Global-Local Framework for Blind Image Quality Assessment"**.

## 1. Overview

Evaluating the perceptual quality of User-Generated Content (UGC) remains highly challenging due to the intricate interplay between diverse authentic distortions and rich semantic content. To bridge the gap between low-level feature statistics and higher-level cognitive processes, **SAGF-IQA** explicitly models the key attention mechanisms of the Human Visual System (HVS) through a synergistic CNN-Transformer architecture. 

By leveraging a frozen $U^2$-Net saliency detector, the framework localizes initial visual fixation regions to generate fine-grained local crops, which are processed in parallel with the global image across multiple hierarchical scales. Heterogeneous features are then adaptively aggregated using a data-driven dynamic fusion module to predict a perceptually aligned quality score.

### 📂 Core Components

* **`SAGF.py`**: The core model definition file. It implements the multi-branch feature extraction architecture (combining dual-branch ResNet50, Swin Transformer Base, and EfficientNetV2-S) alongside the proposed **Channel-Space-Angle (CSA) attention** mechanism and the **Dynamic Multi-branch Feature Fusion (DMFF)** strategy.
* **`IQADataset.py`**: The data loading and preprocessing module. It contains the custom `IQA_dataloader` class, which handles mainstream authentic distortion benchmarks (CLIVE, SPAQ, KonIQ-10k) and manages the synchronized loading of full-resolution images paired with their saliency-cropped patches.
* **`DatabaseCROPⅢ.py`**: The saliency-guided region cropping script. It utilizes the pre-trained $U^2$-Net model to detect human fixation areas on images and automatically generates the aligned $224 \times 224$ cropped patches.
* **`Train.py`**: The main execution script for training. It controls the whole optimization pipeline, including loss function definition (MSE), Adam optimizer setups, and training/validation loops.
* **`utils.py`**: The evaluation metric tool file. It implements the standard mathematical formulas to compute **SRCC** (Spearman Rank-order Correlation Coefficient) and **PLCC** (Pearson Linear Correlation Coefficient) to measure prediction alignment.
* **`u2net.py`**: Contains the network architecture definition of $U^2$-Net, serving as the foundational backbone for saliency map generation.

## 2. Datasets

The following authentic distortion datasets are supported for training and testing:

- **CLIVE Dataset**:
  - **Description**: Real-world images collected from diverse mobile devices under wild environmental conditions.
  - **Dataset URL**: [CLIVE Dataset](https://live.ece.utexas.edu/research/ChallengeDB/index.html)

- **SPAQ Dataset**:
  - **Description**: 11,125 real-world photos focusing on casual user mobile photography.
  - **Dataset URL**: [SPAQ Dataset](https://github.com/h4nwei/SPAQ)

- **KonIQ-10K Dataset**:
  - **Description**: A massive ecologically valid UGC database with large-scale crowdsourced subjective ratings.
  - **Dataset URL**: [KonIQ-10K Dataset](http://database.mmsp-kn.de/koniq-10k-database.html)

### Data Preparation:
- Download the raw images from the official links provided above.
- Ensure the image folders and their corresponding CSV files (containing Mean Opinion Scores / MOS annotations) are properly organized in your local path.

## 3. Installation

### Environment Dependencies
We provide a comprehensive `requirements.txt` file containing all the necessary packages for this framework. You can automatically configure and install the entire environment with a single command:

```bash
pip install -r requirements.txt
