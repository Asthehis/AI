# Deep Learning MLOps with PyTorch - Course Materials
## From Data to Production: Complete MLOps Pipeline

---

## 📚 Course Structure

**Format**: 3 sessions × 8 hours = 24 hours total  
**Group**: 3 students  
**Datasets**: Pneumonia Detection (X-Ray) OR Solar Panel Classification  
**Platform**: Google Colab (GPU T4 free tier sufficient)

---

## 📂 Repository Structure

```
Deep_Learning_MLOps_Course/
├── README.md        
├── Deep_Learning_Course_Introduction.md # Complete course intro                   # This file
├── utils.py                            # Helper functions (ALL themes)
├── notebooks/
│   ├── theme1_data_analysis.md         # Data exploration & DataLoader
│   ├── theme2_baseline_model.md        # CNN baseline + training loop
│   ├── theme3_optimization.md          # Architecture search & tuning
│   └── themes4to8_deployment_mlops.md  # ONNX, Monitoring, Drift, Retraining, Synthesis


```

---

## 🎯 Learning Objectives

By the end of this course, you will be able to:

1. ✅ **Build CNN architectures** from scratch (PyTorch)
2. ✅ **Optimize training** (mixed precision, schedulers, augmentation)
3. ✅ **Deploy models** with ONNX Runtime (3-10× speedup)
4. ✅ **Monitor production** models (TensorBoard + logging)
5. ✅ **Detect data drift** (statistical tests + embeddings)
6. ✅ **Automate retraining** on drift detection
7. ✅ **Apply MLOps best practices** end-to-end

---

## 📖 8 Themes Overview

### **Theme 1: Data Analysis** (2-3h)
- ✅ Dataset download (Kaggle API)
- ✅ Exploration & statistics
- ✅ DataLoader optimization
- ✅ Data augmentation (domain-specific)
- ✅ Baseline metrics collection

**Deliverable**: `baseline_metrics.pt` file

---

### **Theme 2: Baseline Model** (3-4h)
- ✅ Simple CNN architecture (~1-2M params)
- ✅ Training loop + validation
- ✅ TensorBoard tracking
- ✅ Model Card documentation
- ✅ **Target**: >70% test accuracy

**Deliverable**: `best_model.pth` checkpoint

---

### **Theme 3: Optimization** (3-4h)
- ✅ Residual connections (ResNet-inspired)
- ✅ Mixed precision training (FP16)
- ✅ Learning rate finder
- ✅ Advanced augmentation
- ✅ **Target**: >80% test accuracy

**Deliverable**: `optimized_best.pth` checkpoint

---

### **Theme 4: ONNX & Deployment** (3-4h)
- ✅ PyTorch → ONNX export
- ✅ ONNX Runtime benchmarking
- ✅ **TensorRT (optional bonus)** - may fail on Colab
- ✅ **Target**: 3-10× speedup with <1% accuracy drop

**Deliverable**: `model.onnx` file + benchmark results

---

### **Theme 5: Monitoring** (2-3h)
- ✅ Inference logging (CSV)
- ✅ TensorBoard dashboard
- ✅ Alerting rules
- ✅ Baseline production metrics

**Deliverable**: `production_logs.csv` + TensorBoard screenshots

---

### **Theme 6: Drift Detection** (2-3h)
- ✅ Simulate drift (blur + noise)
- ✅ KS-test (pixel distributions)
- ✅ MMD (embedding drift)
- ✅ Trigger retraining decision

**Deliverable**: Drift detection report

---

### **Theme 7: Retraining** (2-3h)
- ✅ Data mixing (90% baseline + 10% drift)
- ✅ Fine-tuning pipeline
- ✅ Validation gate (no regression)
- ✅ Deploy retrained model

**Deliverable**: `model_retrained.pth`

---

### **Theme 8: Synthesis** (2h) **[MANDATORY]**
- ✅ Performance summary table
- ✅ "When to Use What" framework
- ✅ Lessons learned (worked/didn't/surprises)
- ✅ Best practices
- ✅ MLOps maturity assessment

**Deliverable**: Final report synthesis section

---

## 🚀 Quick Start

### 1. Setup Google Colab

```python
# Install dependencies
!pip install torch torchvision matplotlib seaborn pandas scikit-learn

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Setup working directory
import os
WORK_DIR = '/content/drive/MyDrive/MLOps_Project'
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)

# Clone utils
!wget https://raw.githubusercontent.com/.../utils.py
```

### 2. Configure Kaggle API

```python
# Upload kaggle.json
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

### 3. Choose Dataset

```python
# Option A: Pneumonia
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip -q chest-xray-pneumonia.zip

# Option B: Solar Panels
!kaggle datasets download -d tunguz/solar-panel-classification
!unzip -q solar-panel-classification.zip
```

### 4. Import Utilities

```python
from utils import (
    train_epoch,
    validate,
    detect_drift,
    benchmark_model,
    InferenceLogger,
    set_seed
)

set_seed(42)  # Reproducibility
```

---

## 📊 Expected Results

### Performance Targets

| Stage | Metric | Target | Notes |
|-------|--------|--------|-------|
| **Theme 2: Baseline** | Test Accuracy | >70% | Simple CNN |
| **Theme 3: Optimized** | Test Accuracy | >80% | ResBlocks + AMP |
| **Theme 4: ONNX** | Speedup | >3× | vs PyTorch FP32 |
| **Theme 4: Accuracy** | Degradation | <1% | Post-deployment |
| **Theme 6: Drift** | Detection | Yes | KS-test p<0.05 |
| **Theme 7: Retrained** | Accuracy | Recover | On drifted data |

### Timeline

| Session | Themes | Duration | Deliverables |
|---------|--------|----------|--------------|
| **Session 1** | 1, 2 | 8h | DataLoader + Baseline (>70%) |
| **Session 2** | 3, 4 | 8h | Optimized (>80%) + ONNX |
| **Session 3** | 5, 6, 7, 8 | 8h | Monitoring + Drift + Retrain + Report |

---

## 🛠️ Utilities Reference

### Training

```python
# Basic training
train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
val_loss, val_acc = validate(model, valloader, criterion, device)

# Mixed precision (Theme 3)
from torch.cuda.amp import GradScaler
scaler = GradScaler()
train_loss, train_acc = train_epoch_amp(model, trainloader, criterion, optimizer, scaler, device)
```

### Benchmarking

```python
# PyTorch model
metrics = benchmark_model(model, input_shape=(1, 3, 224, 224), device='cuda')
print(f"Latency P50: {metrics['latency_p50']:.2f}ms")

# ONNX model
metrics = benchmark_onnx('model.onnx', input_shape=(1, 3, 224, 224))
```

### Drift Detection

```python
# Pixel-level drift
drift_detected, p_value = detect_pixel_drift(baseline_loader, production_loader)

# Embedding drift
drift_detected, mmd_score = detect_embedding_drift(
    model, baseline_loader, production_loader, device, threshold=0.3
)
```

### Logging

```python
# Production inference logging
logger = InferenceLogger('production_logs.csv')
logger.log(pred_class=1, confidence=0.95, latency_ms=8.5)
```

---

## 📝 Report Structure

Your final report (50-100 pages) should include:

### Required Sections

1. **Executive Summary** (1-2 pages)
   - Problem & approach
   - Key results quantified
   - Main recommendations

2. **Introduction** (2-3 pages)
   - Context MLOps
   - Dataset chosen (Pneumonia or Solar)
   - Methodology

3. **Theme 1: Data Analysis** (5-8 pages)
   - Statistics, visualizations
   - DataLoader config
   - Augmentation strategy

4. **Theme 2: Baseline Model** (6-8 pages)
   - Architecture
   - Training loop
   - Performance (>70%)

5. **Theme 3: Optimization** (7-10 pages)
   - Architecture improvements
   - Hyperparameter tuning
   - Performance (>80%)

6. **Theme 4: Deployment** (7-10 pages)
   - ONNX export
   - Benchmarking (speedup)
   - Accuracy validation

7. **Theme 5: Monitoring** (5-7 pages)
   - Logging pipeline
   - Dashboard setup
   - Baseline production metrics

8. **Theme 6: Drift Detection** (6-8 pages)
   - Drift simulation
   - Detection methods (2+)
   - Trigger decision

9. **Theme 7: Retraining** (6-8 pages)
   - Data mixing
   - Fine-tuning
   - Validation gate

10. **Theme 8: Synthesis** (6-10 pages) **[MANDATORY]**
    - Performance summary table
    - "When to Use What"
    - Lessons learned
    - Best practices

11. **Conclusion** (2-3 pages)
    - Key learnings
    - Limitations
    - Future work

12. **References**
    - Papers (ResNet, mixed precision, etc.)
    - Documentation (PyTorch, ONNX, etc.)

---

## ✅ Evaluation Criteria (/20)

1. **Comprehension & Completion** (/4)
   - All 8 themes covered
   - Structure complete
   - MLOps pipeline coherent

2. **Technical Quality** (/6)
   - Code functional & reproductible
   - ONNX/TensorRT working
   - Monitoring operational
   - Drift detection implemented

3. **Analysis & Interpretation** (/5)
   - Quantitative metrics
   - Rigorous comparisons
   - Trade-offs analyzed

4. **Presentation Quality** (/3)
   - Clear writing
   - Quality visualizations
   - Well-documented code

5. **Critical Thinking** (/2)
   - Limitations discussed
   - Best practices identified
   - Justified recommendations

### Bonus Points (+2 max)

- **+1**: Interactive demo (Gradio/Streamlit)
- **+1**: CI/CD pipeline (GitHub Actions)
- **+0.5**: Public GitHub repository
- **+0.5**: Blog post

---

## 🐛 Common Issues & Solutions

### Issue: CUDA out of memory

```python
# Solution: Reduce batch size
BATCH_SIZE = 64  # instead of 128
```

### Issue: TensorRT fails on Colab

```python
# Solution: Use ONNX Runtime (acceptable fallback)
print("⚠️ TensorRT failed → Using ONNX Runtime")
# Explain in report why TensorRT failed
```

### Issue: DataLoader slow

```python
# Solution: Adjust num_workers
trainloader = DataLoader(..., num_workers=0)  # Try 0 on Colab if errors
```

### Issue: Model not converging

```python
# Check:
# 1. Learning rate (try 1e-4 to 1e-2)
# 2. Data normalization (mean/std correct?)
# 3. Weight decay (reduce if underfitting)
```

---

## 📚 Resources

### Documentation

- **PyTorch**: https://pytorch.org/docs
- **ONNX**: https://onnx.ai/
- **TensorBoard**: https://www.tensorflow.org/tensorboard

### Papers

- **ResNet**: He et al., "Deep Residual Learning" (2015)
- **Mixed Precision**: Micikevicius et al. (2018)
- **Data Drift**: Rabanser et al., "Failing Loudly" (2019)

### Tutorials

- **PyTorch Tutorials**: pytorch.org/tutorials
- **ONNX Runtime**: onnxruntime.ai/docs
- **TensorRT**: docs.nvidia.com/deeplearning/tensorrt

---

## 🎓 Tips for Success

1. **Start Early**: Theme 1 is foundation for all others
2. **Fix Seeds**: Reproducibility is critical
3. **Save Often**: Use Google Drive for checkpoints
4. **Document As You Go**: Don't wait until end to write report
5. **Ask Questions**: Professor available for technical issues
6. **Test on Small Data First**: Debug faster
7. **Team Communication**: Daily standups recommended

---

**Good luck! 🚀**

*This course will teach you production-ready MLOps, not just model training.*
