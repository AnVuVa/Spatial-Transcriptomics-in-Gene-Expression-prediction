# Spatial Transcriptomics Gene Expression Prediction

## Project Overview

This project focuses on **spatial transcriptomics gene expression prediction** using deep learning approaches. The primary implementation is **BG-TRIPLEX** (Bimodal Guided TRIPLEX), a sophisticated model that integrates histological images with spatial information to predict gene expression patterns in tissue samples.

**Collaborators**: HUST × VinUni

### Key Research Papers

The `papers` directory contains reference papers related to spatial transcriptomics methods:
- **BG-TRIPLEX** - bibm (Primary implementation)
- **CeiTEA** - Advanced Science
- **DANet** - bib
- **ENLIGHT-DeepPT** - Nature Cancer
- **GHIST** - Nature Methods
- **HER2ST** - Dataset reference
- **MCGAE** - bib
- **STANCE** - Nature Communications
- **StereoMM** - bib

## Architecture Overview

### BG-TRIPLEX Model

BG-TRIPLEX is a transformer-based deep learning model that predicts gene expression from histological images by combining multiple modalities:

#### Core Components

1. **Feature Extractors**
   - **Target Feature Extractor**: Processes target spot images using pretrained ResNet18
   - **Neighbor Feature Extractor**: Extracts features from neighboring tissue regions
   - **Guided Feature Encoder**: Processes edge and nuclei segmentation information

2. **Encoder Modules** (`models/module.py`)
   - **GlobalEncoder**: Processes global tissue context using Position-Encoded Graph Hierarchy (PEGH)
   - **NeighborEncoder**: Handles spatial neighbor information with cross-attention mechanisms
   - **FusionEncoder**: Combines multiple modalities (target, neighbor, global) for final prediction

3. **Transformer Components**
   - **MultiHeadAttention**: Self-attention with optional attention bias for spatial relationships
   - **MultiHeadCrossAttention**: Cross-attention between different feature types
   - **TransformerEncoder**: Stacked transformer layers with layer normalization
   - **CrossEncoder**: Cross-attention encoder for multi-modal fusion

#### Model Architecture Details

```
Input: Tissue Image Patches + Edge Detection + Nuclei Segmentation
  └── Target Spot (224×224)
  └── Neighbor Regions (5×5 grid)
  └── Global Context (All spots in tissue)

Processing Pipeline:
1. Feature Extraction (ResNet18 backbone)
   ├── Target patch features
   ├── Edge-guided features
   └── Nuclei-guided features

2. Neighbor Processing
   └── NeighborEncoder (depth2 layers, 8 heads)

3. Global Context
   └── GlobalEncoder (depth3 layers, 4 heads, PEGH positioning)

4. Multi-Modal Fusion
   └── FusionEncoder (depth1 layers, 4 heads)
       ├── Target + Neighbor fusion
       └── Global context integration

Output: Gene Expression Predictions (250 genes)
```

#### Key Parameters

- **Embedding dimension**: 512
- **Transformer depths**: 
  - Fusion: 1 layer (depth1)
  - Neighbor: 5 layers (depth2)
  - Global: 4 layers (depth3)
- **Attention heads**: 4-8 heads per layer
- **MLP ratio**: 4× expansion
- **Dropout rates**: 0.3-0.4
- **Patch size**: 224×224 pixels
- **Neighbor grid**: 5×5 regions
- **Predicted genes**: 250

## Dataset Structure

### STDataset (`datasets/st_data.py`)

The dataset class handles spatial transcriptomics data with multiple modalities:

#### Data Format

```
data/
├── her2st/                          # HER2ST dataset
│   ├── ST-imgs/                     # Histology images
│   ├── ST-spotfiles/                # Spot coordinates (.tsv)
│   ├── gen_per_patch/               # Single spot patches
│   ├── gen_n_patches/               # Neighbor region patches
│   ├── gen_nuc_per_patch/          # Nuclei segmentation patches
│   ├── gen_edge_per_patch/         # Edge detection patches
│   ├── gt_features_224/            # Global target features
│   └── gn_features/                # Global neighbor features
├── test/                           # Test datasets
│   ├── 10x_breast_ff1/
│   ├── 10x_breast_ff2/
│   └── 10x_breast_ff3/
```

#### Supported Datasets

1. **her2st** - Primary training dataset (samples A-H)
2. **stnet** - Alternative training dataset
3. **skin** - Skin tissue samples
4. **visium** - 10x Visium datasets (test sets)

### Data Preprocessing

The `get_patch_csv.py` script extracts patches from whole slide images:

**Features**:
- Extracts 224×224 patches for individual spots
- Generates 5×5 neighbor region patches (1120×1120)
- Processes histology images, nuclei segmentation, and edge detection
- Creates CSV files mapping coordinates to patch files
- Uses `pyvips` for efficient large image processing

**Usage** (interactive script):
1. Select dataset type (her2st/stnet/skin/visium)
2. Automatically processes all samples in the dataset
3. Generates:
   - Individual patch images
   - Neighbor patch grids
   - Nuclei segmentation patches
   - Edge detection patches
   - CSV mapping files

## Training Pipeline

### Configuration (`config/her2st/BG_TRIPLEX.yaml`)

```yaml
GENERAL:
  seed: 2021
  log_path: ./logs

TRAINING:
  num_k: 1                    # Number of cross-validation folds
  batch_size: 1
  optimizer: adam
  learning_rate: 1.0e-4
  num_epochs: 200
  
  early_stopping:
    monitor: R                # Pearson correlation
    patience: 20
    mode: max

MODEL:
  name: BG_TRIPLEX
  num_genes: 250
  emb_dim: 512
  # ... (transformer parameters)

DATASET:
  data_dir: data
  type: her2st
  num_neighbors: 5
  radius: 224
```

### Training Modes

#### 1. Cross-Validation (`mode='cv'`)
```bash
python main.py --config_name her2st/BG_TRIPLEX --mode cv --gpu 0
```

#### 2. Testing (`mode='test'`)
```bash
python main.py --config_name her2st/BG_TRIPLEX --mode test \
  --model_path logs/checkpoint.ckpt --gpu 0
```

#### 3. External Testing (`mode='external_test'`)
```bash
python main.py --config_name her2st/BG_TRIPLEX --mode external_test \
  --test_name 10x_breast_ff1 --model_path logs/checkpoint.ckpt --gpu 0
```

#### 4. Inference (`mode='inference'`)
Generates predictions and embeddings for downstream analysis:
```bash
python main.py --config_name her2st/BG_TRIPLEX --mode inference \
  --test_name 10x_breast_ff1 --model_path logs/checkpoint.ckpt --gpu 0
```

### Training Framework

- **Framework**: PyTorch Lightning 1.8.0
- **Strategy**: DDP (Distributed Data Parallel)
- **Precision**: Mixed precision training (FP16)
- **Logging**: TensorBoard + CSV Logger
- **Callbacks**: 
  - Early stopping (monitors Pearson R)
  - Model checkpointing (saves best model)

## Evaluation Metrics

The model is evaluated using multiple metrics:

1. **MSE (Mean Squared Error)**: Loss between predicted and actual gene expression
2. **MAE (Mean Absolute Error)**: Absolute difference metric
3. **Pearson Correlation (R)**: Correlation between predictions and ground truth across genes
4. **Per-spot metrics**: Calculated for each spatial location in tissue

## Utility Functions (`utils.py`)

### Data Processing

- **`match(x, y)`**: Matches spot IDs between position and expression data
- **`smooth_exp(cnt)`**: Applies spatial smoothing by averaging 9-neighbor spots
- **`collate_fn(batch)`**: Custom batch collation for DataLoader

### Configuration & Logging

- **`load_config(config_name)`**: Loads YAML configuration files
- **`load_loggers(cfg)`**: Initializes TensorBoard and CSV loggers
- **`load_callbacks(cfg)`**: Sets up early stopping and checkpointing

## Requirements

### Core Dependencies

```
PyTorch
pytorch-lightning==1.8.0.post1
torchmetrics==0.10.2
einops
```

### Data Processing

```
numpy
pandas
scikit-learn
scipy
h5py
pyarrow
```

### Imaging & Visualization

```
Pillow
pyvips
tifffile==2022.10.10
scanpy
scprep
```

### Utilities

```
PyYAML
addict
tqdm
easydl
```

## Project Structure

```
git_clone/BG-TRIPLEX/
├── main.py                  # Training/testing entry point
├── config/                  # Configuration files
│   └── her2st/
│       └── BG_TRIPLEX.yaml
├── models/                  # Model implementations
│   ├── BG_TRIPLEX.py       # Main model class
│   ├── module.py           # Encoder modules
│   └── __init__.py
├── datasets/                # Dataset loaders
│   ├── st_data.py          # Spatial transcriptomics dataset
│   └── __init__.py
├── utils.py                # Utility functions
├── get_patch_csv.py        # Patch extraction script
└── requirements.txt        # Python dependencies
```

## Key Features

### 1. Multi-Modal Integration
- Combines histology images, edge detection, and nuclei segmentation
- Leverages spatial neighborhood information
- Integrates global tissue context

### 2. Hierarchical Processing
- Local target spot features
- Neighboring region context (5×5 grid)
- Global tissue-wide relationships

### 3. Knowledge Transfer Training
The model uses a unique training strategy that transfers knowledge from the fusion module to individual branches:
- Fusion module combines all modalities
- Individual branches (target, neighbor, global) are guided by fusion predictions
- Enables robust multi-scale gene expression prediction

### 4. Flexible Inference
- Supports multiple testing modes
- Can generate embeddings for downstream analysis
- Handles external datasets not seen during training

## Usage Workflow

### 1. Data Preparation
```bash
# Run interactive patch extraction
python get_patch_csv.py
# Select dataset: 1 (her2st), 2 (stnet), 3 (skin), or 4 (visium)
```

### 2. Training
```bash
# Configure parameters in config/her2st/BG_TRIPLEX.yaml
python main.py --config_name her2st/BG_TRIPLEX --mode cv --gpu 0
```

### 3. Evaluation
```bash
# Test on external dataset
python main.py --config_name her2st/BG_TRIPLEX \
  --mode external_test \
  --test_name 10x_breast_ff1 \
  --model_path logs/2024-04-10/checkpoint.ckpt \
  --gpu 0
```

### 4. Inference
```bash
# Generate predictions and embeddings
python main.py --config_name her2st/BG_TRIPLEX \
  --mode inference \
  --test_name 10x_breast_ff1 \
  --model_path logs/checkpoint.ckpt \
  --fold 0 \
  --gpu 0
```

## Model Outputs

### Training/Testing
- Validation metrics logged per epoch
- Best model checkpoint saved based on Pearson R
- TensorBoard visualizations

### Inference Mode
- **Predictions**: Gene expression predictions saved as `.npy` files
- **Embeddings**: Feature embeddings saved for downstream analysis
- Output directory: `data/test/{test_name}/pred_{fold}` and `emb_{fold}`

## Advanced Features

### Position-Encoded Graph Hierarchy (PEGH)
A novel positional encoding mechanism that:
- Uses depthwise convolution for position encoding
- Maintains spatial relationships in tissue
- Enables the model to understand tissue topology

### Attention Mechanisms
- **Self-attention**: Within each modality (target, neighbor, global)
- **Cross-attention**: Between modalities for fusion
- **Masked attention**: Handles variable-sized neighbor regions
- **Attention bias**: Optional learned bias for spatial relationships

### Custom Prediction Writer
The `CustomWriter` class enables batch prediction saving during inference:
- Saves predictions and embeddings per sample
- Handles variable-length outputs
- Organizes outputs by fold and sample name

## Citation

If using this codebase, please refer to the BG-TRIPLEX paper in the `papers` directory.

---

*Last updated: 2025-12-19*
