# IEF-MIL
**Instance and Embedding Fused Multiple Instance Learning**

![Model Architecture](docs/mamo3.png)

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Whole Slide Image Preprocessing](#usage)
  - [Segmentation and Patching](#segmentation-and-patching)
  - [Feature Extraction](#feature-extraction)
- [Training](#training)
- [Testing](#testing)
- [Multi-Scale Inference](#multi-scale-inference)

## Acknowledgements
This repository is sourced from [CLAM](https://github.com/mahmoodlab/CLAM). The CLAM repository provided foundational work and inspiration for the development of this project. We thank the authors for their contributions to the field.

## Overview
This project implements Instance and Embedding Fused Multiple Instance Learning (IEF-MIL) for whole slide image analysis. The primary functionalities include segmentation, patching, and feature extraction.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repo-name.git

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Whole Slide Image Preprocessing

1. Segmentation and Patching:
   ```bash
   python create_patches_fp.py --source path/to/data_dir --save_dir path/to/save_dir --patch_size 224 --seg --patch --stitch

--source: Path to the directory containing the whole slide images.
--save_dir: Directory where the generated patches will be saved.
--patch_size: Size of the patches to be created (e.g., 224).
--seg: Optional flag for segmentation.
--patch: Optional flag to create patches.
--stitch: Optional flag to stitch the patches back together.

2. Feature Extraction
   ```bash
   python extract_features_fp.py --data_h5_dir path/to/features_dir --data_slide_dir path/to/data_dir --csv_path path/to/features_dir/process_list_autogen.csv --feat_dir path/to/features_dir --batch_size 224 --slide_ext .ndpi

--data_h5_dir: Directory where the extracted features will be saved.
--data_slide_dir: Directory containing the whole slide images.
--csv_path: Path to the CSV file listing the images for feature extraction.
--feat_dir: Directory for storing the extracted features.
--batch_size: Number of images to process in each batch (e.g., 224).
--slide_ext: File extension of the slide images (e.g., .ndpi).

## Training
Detailed instructions for training models will be provided here. Currently, refer to the scripts and configuration files for information on training procedures.

## Testing
Information on how to test models and evaluate performance will be added here. Please check the relevant scripts and configuration files for testing guidelines.
Used metrics: AUC-ROC, AUC-PR, per class PR curves, F1 score and accuracy

## Multi-Scale Inference
```bash
python create_heatmaps.py --config config_template.yaml
