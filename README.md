# ProtoAF: An Explainable and Lightweight Prototype-Based Framework for Cross-Domain Atrial Fibrillation Screening

## Introduction
This repository is the official implementation of **ProtoAF**, a lightweight and explainable prototype-based framework designed for cross-domain atrial fibrillation (AF) screening. 

Different from conventional black-box deep learning models, ProtoAF introduces prototype learning to enhance model interpretability while maintaining a lightweight architecture. It achieves stable and robust atrial fibrillation detection under cross-domain scenarios, effectively alleviating domain shift in clinical ECG data.

## Project Structure

```
├── checkpoints          # Saved model weights and training checkpoints
├── config               # YAML configuration files (paths, hyperparameters)
├── dataset              # Dataset loader and DataLoader definition
├── dataset_preprocess   # Raw ECG data preprocessing scripts
├── dataset_url          # Official dataset download links
├── model                # ProtoAF core network implementation
├── results              # Experimental logs, metrics and visualization results
├── train.py             # Training entry
└── test.py              # Evaluation and inference entry
```

## Environment Requirements
```
pip install torch torchvision numpy pandas scikit-learn pyyaml matplotlib tqdm
```

## Quick Start

### 1. Download Dataset

Get public ECG dataset download links from the `dataset_url` folder and download the raw data.

### 2. Modify Configuration

Update the **dataset path** in the `.yaml` configuration file under the `config` folder to your local dataset directory.

### 3. Model Training

```
python train.py
```

Checkpoints will be automatically saved in `checkpoints/`, and training logs will be saved in `results/`.

### 4. Model Testing

```
python test.py
```

Evaluate the cross-domain AF screening performance and output quantitative metrics.

## Framework Features

- **Lightweight**: Low computational cost, suitable for clinical edge deployment.
- **Explainable**: Prototype-based visual interpretation for ECG classification decisions.
- **Cross-domain Robust**: Effectively reduces domain gap across different ECG datasets.
