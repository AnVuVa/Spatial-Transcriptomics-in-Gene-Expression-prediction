# BG-TRIPLEX Research Analysis

## Paper: BG-TRIPLEX (Bimodal Guided TRIPLEX)
**Conference**: IEEE BIBM (Bioinformatics and Biomedicine)  
**Focus**: Spatial Transcriptomics Gene Expression Prediction

---

## Research Overview

### Problem Statement
Predicting gene expression patterns from histological images in spatial transcriptomics data requires:
1. **Spatial context understanding**: Genes are influenced by surrounding tissue
2. **Multi-scale information**: Local features + neighbor context + global tissue patterns
3. **Histological guidance**: Edge detection and nuclei segmentation provide structural information

### BG-TRIPLEX Solution
A **transformer-based deep learning model** that integrates:
- **Target spot** features (local)
- **Neighbor region** features (spatial context)  
- **Global tissue** features (tissue-wide patterns)
- **Bimodal guidance**: Edge detection + nuclei segmentation

---

## Architecture Analysis

### Core Innovation: Bimodal Guided Enhancement

Based on the code analysis, BG-TRIPLEX introduces **guided feature encoding** using two modalities:

```python
# From BG_TRIPLEX.py lines 119-121
self.neighbor_feature_extractor = NeighborFeatureExtractor()
self.target_feature_extractor = TargetFeatureExtractor()
self.guided_feature_encoder = GuidedFeatureEncoder(emb_dim, depth1, num_heads1, 
                                int(emb_dim * mlp_ratio1), dropout1)
```

**Bimodal Guidance** means using **two additional image modalities**:
1. **Edge features** (x_edge, x_n_edge): Detect tissue boundaries and structures
2. **Segmentation features** (x_seg, x_n_seg): Identify nuclei and cellular components

### Three-Level Hierarchical Processing

#### 1. **Target Encoder** (Local Features)
```python
# Extract features from target spot (224×224)
target_token = self.target_encoder(x)  # ResNet18 backbone → 512×7×7
target_token = rearrange(target_token, 'b d h w -> b (h w) d')  # → B×49×512

# Apply bimodal guidance for target
edge_features = self.target_feature_extractor(x_edge)
seg_features = self.target_feature_extractor(x_seg)
target_token = self.guided_feature_encoder(edge_features, seg_features, target_token)
```

**Purpose**: Extract fine-grained features from the target tissue spot, enhanced by edge and nuclei information.

#### 2. **Neighbor Encoder** (Spatial Context)
```python
# Process neighbor features (5×5 grid)
neighbor_token = self.neighbor_encoder(neighbor, mask)
neighbor_token = F.relu(neighbor_token)

# Apply bimodal guidance for neighbors
neighbor_edge_features = self.neighbor_feature_extractor(x_n_edge)
neighbor_seg_features = self.neighbor_feature_extractor(x_n_seg)
neighbor_token = self.guided_feature_encoder(neighbor_edge_features, 
                                             neighbor_seg_features, neighbor_token)
```

**Components**:
- **TransformerEncoder** with attention bias for spatial relationships
- **Resolution**: 5×5 neighbor grid
- **Masked attention**: Handles variable neighbor counts at tissue boundaries

**Purpose**: Capture spatial dependencies between the target spot and surrounding regions.

#### 3. **Global Encoder** (Tissue-Wide Context)
```python
# Process all spots in the tissue
global_token = self.global_encoder(x_total, position)

# Uses PEGH (Position-Encoded Graph Hierarchy)
class PEGH:
    def forward(self, x, pos):
        # Create sparse tensor from positions
        x_sparse = torch.sparse_coo_tensor(pos.T, x.squeeze())
        x_dense = x_sparse.to_dense().permute(2,1,0).unsqueeze(0)
        
        # Apply depthwise convolution for position encoding
        x_pos = self.proj1(x_dense)  # Conv2d with groups=dim
        x_out = x_pos_sparse.values().unsqueeze(0)
        return x_out
```

**Purpose**: Model tissue-wide gene expression patterns using positional information.

#### 4. **Fusion Encoder** (Multi-Modal Integration)
```python
class FusionEncoder:
    def forward(self, x_t, x_n, x_g, mask=None):
        # Fuse global with target via cross-attention
        fus1 = self.fusion_layer(x_g.unsqueeze(1), x_t)
        
        # Fuse global with neighbor via cross-attention
        fus2 = self.fusion_layer(x_g.unsqueeze(1), x_n, mask=mask)
        
        # Combine fusions
        fusion = (fus1 + fus2).squeeze(1)
        return self.norm(fusion)
```

**Purpose**: Integrate target, neighbor, and global information through cross-attention.

---

## Training Strategy: Knowledge Distillation

### Multi-Branch Loss Function

The model uses **knowledge transfer from fusion to individual branches**:

```python
def training_step(self, batch, batch_idx):
    outputs = self(...)  # Returns: fusion, initial_target, target, 
                         #          initial_neighbor, neighbor, global
    
    # Fusion loss (main supervision)
    loss = F.mse_loss(outputs[0], exp)
    
    # Target branch: Supervised + Distillation
    loss += F.mse_loss(outputs[2], exp) * (1 - alpha)  # Supervised
    loss += F.mse_loss(outputs[0], outputs[2]) * alpha # Distillation
    
    # Neighbor branch: Supervised + Distillation  
    loss += F.mse_loss(outputs[4], exp) * (1 - alpha)
    loss += F.mse_loss(outputs[0], outputs[4]) * alpha
    
    # Global branch: Supervised + Distillation
    loss += F.mse_loss(outputs[5], exp) * (1 - alpha)
    loss += F.mse_loss(outputs[0], outputs[5]) * alpha
```

**Key Parameters**:
- **alpha = 0.3**: Balances supervised learning vs distillation
- Each branch learns from:
  - Ground truth gene expression (70%)
  - Fusion module predictions (30%)

**Benefits**:
1. Fusion module learns from all modalities
2. Individual branches become robust through guidance
3. Enables multi-scale predictions

---

## Technical Implementation Details

### Transformer Components

#### Multi-Head Attention with Spatial Bias
```python
class MultiHeadAttention:
    def __init__(self, emb_dim, heads=4, attn_bias=False, resolution=(5,5)):
        if attn_bias:
            # Compute pairwise distance-based attention bias
            points = list(itertools.product(range(resolution[0]), range(resolution[1])))
            # Create bias based on spatial offsets
            attention_offsets = {}
            for p1 in points:
                for p2 in points:
                    offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                    # Learn bias for each unique spatial offset
```

**Purpose**: Encode spatial relationships in the 5×5 neighbor grid.

#### Position-Encoded Graph Hierarchy (PEGH)

Novel positional encoding that:
1. Converts spot coordinates to sparse 2D tensor
2. Applies depthwise convolution for position-aware features
3. Maintains spatial topology of tissue

**Advantages**:
- Handles irregular spatial arrangements
- More flexible than standard positional encodings
- Preserves tissue structure

### Data Processing Pipeline

```
Input Images (224×224 patches)
  ├── Histology image
  ├── Edge detection image  
  └── Nuclei segmentation image

↓ Feature Extraction (ResNet18)

Target Features (512-dim)
  ├── From histology (49 tokens)
  ├── From edge (1 token)
  └── From nuclei (1 token)

↓ Guided Feature Encoding

Enhanced Target Features
  + Neighbor Features (25 tokens, masked)
  + Global Features (N tokens with positions)

↓ Multi-Modal Fusion

Gene Expression Prediction (250 genes)
```

---

## Missing Code Components

Based on the current implementation, these classes need to be implemented:

### 1. **NeighborFeatureExtractor**
Should extract and aggregate features from neighbor edge/segmentation images.

### 2. **TargetFeatureExtractor**  
Should extract features from target edge/segmentation images.

### 3. **GuidedFeatureEncoder**
Should integrate edge and segmentation guidance into main features using cross-attention or additive fusion.

**Expected signature**:
```python
class GuidedFeatureEncoder(nn.Module):
    def __init__(self, emb_dim, depth, heads, mlp_dim, dropout):
        # Cross-attention or fusion mechanism
        
    def forward(self, edge_features, seg_features, main_features):
        # Enhance main_features with edge and seg guidance
        return enhanced_features
```

---

## Evaluation Metrics

### Training Metrics
- **MSE Loss**: Mean squared error between predictions and ground truth
- **Pearson R**: Correlation coefficient (primary evaluation metric)
- **MAE**: Mean absolute error (testing only)

### Cross-Validation
```python
# Configuration: config/her2st/BG_TRIPLEX.yaml
TRAINING:
  num_k: 1  # Number of folds
  early_stopping:
    monitor: R        # Pearson correlation
    patience: 20
    mode: max         # Maximize correlation
```

---

## Key Research Contributions

### 1. **Bimodal Guidance**
First approach to integrate edge detection AND nuclei segmentation for guided gene expression prediction.

### 2. **Knowledge Distillation Strategy**
Transfer learning from fusion module to individual branches ensures robust multi-scale predictions.

### 3. **PEGH Positional Encoding**
Novel position encoding using spatial convolutions adapted for irregular tissue layouts.

### 4. **Three-Level Hierarchy**
Systematically combines:
- Local morphology (target)
- Spatial dependencies (neighbor)
- Global patterns (tissue-wide)

---

## Code-Paper Alignment

| Paper Concept | Code Implementation | File Location |
|--------------|---------------------|---------------|
| Target Module | `target_encoder` (ResNet18) | `BG_TRIPLEX.py:113` |
| Neighbor Module | `NeighborEncoder` | `module.py:279` |
| Global Module | `GlobalEncoder` | `module.py:251` |
| Fusion Module | `FusionEncoder` | `module.py:298` |
| PEGH | `PEGH` class | `module.py:228` |
| Attention Bias | `MultiHeadAttention` | `module.py:37` |
| Bimodal Guidance | `GuidedFeatureEncoder` | **MISSING** |
| Knowledge Transfer | `training_step` | `BG_TRIPLEX.py:249` |

---

## Datasets & Experiments

### Training Dataset
- **HER2ST**: Primary dataset (samples A-H)
  - Samples A-D: 6 sections each
  - Samples E-H: 3 sections each

### Test Datasets
- **10x Visium**: External validation
  - 10x_breast_ff1
  - 10x_breast_ff2  
  - 10x_breast_ff3
- **STNet**: Alternative spatial transcriptomics data
- **Skin**: Skin tissue samples

### Hyperparameters (from config)
```yaml
MODEL:
  emb_dim: 512
  depth1: 1   # Fusion
  depth2: 5   # Neighbor
  depth3: 4   # Global
  num_heads1: 4
  num_heads2: 8
  num_heads3: 4
  dropout1: 0.4
  dropout2: 0.3
  dropout3: 0.3

TRAINING:
  batch_size: 1
  learning_rate: 1.0e-4
  num_epochs: 200
  optimizer: adam
```

---

## Implementation Status

### ✅ Implemented
- [x] Three-level hierarchical architecture
- [x] PEGH positional encoding
- [x] Transformer encoders with attention bias
- [x] Knowledge distillation training
- [x] Multi-dataset support
- [x] Cross-validation framework

### ⚠️ Incomplete
- [ ] **GuidedFeatureEncoder** class
- [ ] **TargetFeatureExtractor** class  
- [ ] **NeighborFeatureExtractor** class

These missing components prevent the code from running properly. The imports reference these classes, but they're not defined in the codebase.

---

## Recommendations for Next Steps

1. **Implement Missing Classes**: Create the three feature extractor/encoder classes
2. **Verify Bimodal Guidance**: Ensure edge and segmentation features properly enhance main features
3. **Test Knowledge Transfer**: Validate that distillation improves individual branch performance
4. **Ablation Studies**: Test contribution of each component (target, neighbor, global, guidance)
5. **Document Model Outputs**: Track which branch performs best for different tissue types

---

*Analysis Date: 2025-12-19*  
*Based on code in: `E:\Spatial-Transcriptomics-in-Gene-Expression-prediction\git_clone\BG-TRIPLEX`*
