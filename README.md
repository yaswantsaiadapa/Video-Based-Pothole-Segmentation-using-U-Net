# 🚧 Video-Based Pothole Segmentation using U-Net

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

This project implements a **complete deep learning pipeline** for pothole detection and segmentation from road surveillance videos. It demonstrates a production-ready workflow combining computer vision, neural networks, and temporal video processing.

### Key Capabilities

✅ **Frame-wise Segmentation** - Pixel-level pothole detection using U-Net  
✅ **Temporal Smoothing** - Reduces flickering between consecutive frames  
✅ **Persistence Logic** - Confirms detections only in stable regions  
✅ **Severity Estimation** - Quantifies pothole severity at video level  
✅ **Robust Evaluation** - Dice (0.649) and IoU (0.535) metrics on test set  

---

## 🎯 Problem Statement

Road potholes are a critical infrastructure challenge:
- **Safety Risk**: Vehicle damage, accidents, traffic disruptions
- **Maintenance Challenge**: Manual inspection is time-consuming and inconsistent
- **Scale Issue**: Thousands of kilometers of roads require monitoring

### Solution Approach

Build an **automated vision system** that:
1. Detects potholes in video frames with high accuracy
2. Produces pixel-wise segmentation masks
3. Filters false positives using temporal consistency
4. Estimates damage severity for priority-based repairs

---

## 🧠 Methodology

### 1️⃣ Dataset Preparation

**Source**: Pothole video dataset with synchronized RGB and mask videos

**Processing Pipeline**:
```
Raw Videos (RGB + Masks)
        ↓
Frame Extraction (FRAME_SKIP = 3)
        ↓
Train / Val / Test Split
        ↓
Binary Mask Conversion
        ↓
Augmented Training Set
```

**Dataset Structure**:
```
extracted_frames/
├── train/     (RGB images + binary masks)
├── val/       (validation split)
└── test/      (evaluation split)
```

### 2️⃣ Data Preprocessing & Augmentation

**Image Processing**:
- Resized to **256×256** for model input
- Normalized to **[0, 1]** range
- Masks converted to **binary format** (0/1)

**Training Augmentations** (applied on-the-fly):
```python
✓ Horizontal Flip (50% probability)
✓ Random Rotation (±10°)
✓ Brightness Adjustment (0.8x - 1.2x)
```

### 3️⃣ Model Architecture: U-Net

Custom U-Net implementation built from scratch in PyTorch with skip connections.

![U-Net Architecture](images/architecture.png)

**Architecture Details**:

| Component | Input Channels | Output Channels | Spatial Size |
|-----------|---------------|-----------------|--------------|
| Input     | -             | 3               | 256×256      |
| Down Block 1 | 3          | 32              | 128×128      |
| Down Block 2 | 32         | 64              | 64×64        |
| Down Block 3 | 64         | 128             | 32×32        |
| Down Block 4 | 128        | 256             | 16×16        |
| **Bottleneck** | 256      | 512             | 8×8          |
| Up Block 4 | 512 + skip   | 256             | 16×16        |
| Up Block 3 | 256 + skip   | 128             | 32×32        |
| Up Block 2 | 128 + skip   | 64              | 64×64        |
| Up Block 1 | 64 + skip    | 32              | 128×128      |
| Output    | 32           | 1 (sigmoid)     | 256×256      |

**Key Features**:
- DoubleConv blocks (2× Conv2d + ReLU per level)
- Skip connections reduce information loss
- Symmetric encoder-decoder structure
- Sigmoid activation for binary probability output

### 4️⃣ Loss Function

**Combined Loss Strategy**:
```
Loss = Binary Cross Entropy + Dice Loss
```

**Why this combination?**
- **BCE**: Pixel-level classification accuracy
- **Dice**: Handles class imbalance (potholes are sparse)
- **Synergy**: Better convergence than either alone

```python
def combined_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice
```

---

## 📊 Quantitative Results

### Frame-wise Evaluation (Raw Model)

Model evaluated on test split with single-frame predictions:

| Metric | Score | Interpretation |
|--------|-------|-----------------|
| **Dice Coefficient** | 0.6491 | Strong overlap between predicted and ground truth masks |
| **IoU (Intersection over Union)** | 0.5353 | 53.5% pixel-level accuracy in pothole regions |

### Temporal Video Segmentation (With Persistence)

After applying temporal smoothing and persistence logic:

| Metric | Score | Trade-off |
|--------|-------|-----------|
| **Dice** | 0.5103 | ↓ 21% (spatial accuracy) |
| **IoU** | 0.3874 | ↓ 27% (spatial accuracy) |
| **Flickering** | Reduced | ✓ Stable detections |
| **False Positives** | Lower | ✓ Fewer false alarms |

**Insight**: Trading spatial accuracy for temporal stability produces **production-ready** detections over short video sequences.

---

## 🎥 Temporal Processing Pipeline

### Stage 1: Temporal Smoothing

**Problem**: Single-frame predictions flicker across frames  
**Solution**: Average predictions over N consecutive frames

```python
N = 5  # Smoothing window size

# Maintain deque of last N probability maps
for frame in video:
    prob_map = model(frame)  # Raw model output [0, 1]
    prob_buffer.append(prob_map)
    
    smoothed = np.mean(prob_buffer, axis=0)  # Average
    mask = (smoothed > 0.5).astype(uint8)    # Threshold
```

**Effect**: Reduces noise while preserving genuine pothole regions

### Stage 2: Persistence Logic

**Problem**: Isolated detections may be false positives  
**Solution**: Confirm detection only if present in K consecutive frames

```python
K = 3  # Require 3 consecutive detections
AREA_THRESHOLD = 200  # Minimum pixels

if current_area > AREA_THRESHOLD:
    consecutive_count += 1
else:
    consecutive_count = max(0, consecutive_count - 1)

if consecutive_count >= K:
    final_mask = valid  # Show detection
else:
    final_mask = zeros  # Suppress false positive
```

**Results**:
- ✓ Eliminates single-frame noise
- ✓ Maintains detection of real potholes
- ✓ Produces stable video output

### Stage 3: Video-Level Severity Estimation

For each test video, compute:

```python
frames_with_pothole = count of frames with detected potholes
visible_duration = frames_with_pothole / fps
mean_area_ratio = average pixel ratio per frame
max_area_ratio = peak pixel coverage

# Classify severity
if mean_area_ratio < 0.03:
    severity = "Small"     # Minor surface damage
elif mean_area_ratio < 0.10:
    severity = "Medium"    # Moderate pothole
else:
    severity = "Large"     # Severe damage
```

---

## 🖼️ Sample Results

### Segmentation Output

![Sample Segmentation Result](images/result_sample.png)

*Left: Original frame | Right: Detected pothole region highlighted in red*

### Video Outputs

Three output formats are generated:

| Output | Feature | File |
|--------|---------|------|
| **Baseline** | Single-frame predictions | `outputs/output_baseline.mp4` |
| **Smoothed** | Temporal averaging applied | `outputs/output_smoothed.mp4` |
| **Persistent** | Full pipeline with persistence | `outputs/output_persistant.mp4` |

---

## 🛠️ Technology Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.8+ | Programming language |
| **PyTorch** | 2.0+ | Deep learning framework |
| **OpenCV** | 4.8+ | Video and image processing |
| **NumPy** | 1.24+ | Numerical computations |
| **Matplotlib** | 3.8+ | Visualization |

---

## 🚀 How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Requires**:
- Python 3.8 or higher
- CUDA-compatible GPU (optional, defaults to CPU)

### Step 2: Prepare Dataset

Place your pothole video dataset in the following structure:

```
pothole_dataset/
└── pothole_video/
    ├── train/
    │   ├── rgb/     (RGB video files)
    │   └── mask/    (Mask video files)
    ├── val/
    │   ├── rgb/
    │   └── mask/
    └── test/
        ├── rgb/
        └── mask/
```

### Step 3: Run the Notebook

```
1. Open project.ipynb in Jupyter
2. Execute cells in order:
   
   Phase 1: Frame extraction from videos
   Phase 2: Dataset creation and loading
   Phase 3: Model training
   Phase 4: Model evaluation
   Phase 5-6: Video inference (baseline)
   Phase 7: Temporal smoothing
   Phase 8: Persistence-based filtering
   Phase 9: Severity estimation
   Phase 10: Final evaluation
```

### Step 4: Generate Outputs

The notebook automatically produces:
- ✓ Trained model weights
- ✓ Segmentation evaluation metrics
- ✓ Output videos with overlaid predictions
- ✓ Severity reports per video

---

## 📁 Project Structure

```
project-pothole-segmentation/
│
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Dependency list
├── 📓 project.ipynb                # Main notebook (all phases)
│
├── 📁 images/
│   ├── architecture.png            # U-Net architecture diagram
│   └── result_sample.png           # Example segmentation result
│
└── 📁 outputs/
    ├── output_baseline.mp4         # Raw model predictions
    ├── output_smoothed.mp4         # Temporal smoothing applied
    └── output_persistant.mp4       # Full pipeline output
```

---

## 🔍 Key Implementation Details

### Custom Dataset Class

```python
class PotHoleDataset(Dataset):
    def __init__(self, rgb_dir, mask_dir, img_size=256, train=False):
        self.rgb_images = sorted(os.listdir(rgb_dir))
        self.train = train
        
    def __getitem__(self, idx):
        # Load and preprocess image
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.tensor(image/255.0, dtype=torch.float32).permute(2,0,1)
        
        # Load and preprocess mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = torch.tensor((mask>0).astype("float32")).unsqueeze(0)
        
        # Apply augmentations (training only)
        if self.train:
            image, mask = self.augment(image, mask)
        
        return image, mask
```

### Training Loop

```python
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        preds = model(images)
        loss = combined_loss(preds, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)
```

---

## 💡 Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **U-Net from scratch** | Custom control, educational value | Less accurate than pretrained backbones |
| **256×256 images** | Memory efficient, faster training | Loses fine detail (larger potholes) |
| **Combined BCE + Dice** | Handles class imbalance well | Increased hyperparameter tuning |
| **N=5 smoother, K=3 persistence** | Good balance for real-time | Misses small/fast potholes |
| **Temporal smoothing first** | Stabilizes predictions | Adds ~5 frame latency in video |

---

## 🏆 Key Achievements

✅ **Functional End-to-End System**: From video input to severity report  
✅ **Robust Evaluation**: Comprehensive metrics beyond accuracy  
✅ **Temporal Intelligence**: Not just frame-wise but video-aware  
✅ **Production Considerations**: Handles real-world noise and false positives  
✅ **Interpretability**: Clear visualization of what model detects  

---

## 📚 Future Improvements

| Priority | Enhancement | Benefit |
|----------|-----------|---------|
| **High** | Pretrained encoder (ResNet-50 backbone) | 15-20% accuracy boost |
| **High** | Adaptive threshold learning | Better generalization |
| **Medium** | Attention mechanisms (CBAM) | Focus on relevant features |
| **Medium** | Use larger dataset | Reduce overfitting |
| **Low** | Real-time inference optimization | Deploy on edge devices |
| **Low** | Multi-class segmentation | Pothole severity classification |

---

## 📖 Lessons & Takeaways

### Machine Learning
- Implementing architectures from scratch deepens understanding
- Loss function design is crucial for task-specific performance
- Class imbalance requires thoughtful metric selection

### Computer Vision
- Temporal consistency matters as much as spatial accuracy
- Simple post-processing (smoothing, persistence) provides huge practical gains
- Visualization is essential for debugging

### Software Engineering
- Modular pipeline design enables easy modifications
- Clear separation: preprocessing → model → post-processing
- Evaluation metrics guide iteration and improvements

---

## 🎓 Author

**Yaswant Sai**  
B.Tech CSE | Deep Learning & Computer Vision Enthusiast

---

## 📝 License

This project is available under the MIT License. See LICENSE file for details.

---

## 🤝 Contributing

Suggestions and improvements are welcome! Feel free to:
- Report issues
- Propose enhancements
- Submit pull requests

---

**Built with ❤️ for road safety and infrastructure maintenance**
