# Pneumonia Detection from Chest X-Ray Images

## Overview

This project implements an automated **pneumonia detection system** using chest X-ray images through both traditional machine learning and deep learning approaches. The system classifies X-ray images into two categories: **NORMAL** and **PNEUMONIA**. This is critical for early diagnosis and treatment, particularly in resource-constrained settings where radiologist availability is limited.

**Key Results:**
- **Best Model:** VGG16 Transfer Learning
- **Test Accuracy:** 85.42%
- **ROC AUC Score:** 92.96%
- **Precision:** 86.73% | **Recall:** 90.51%

---

## Dataset

### Source
Dataset Link: [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).


### Dataset Statistics
| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| **Training** | 1,341 | 3,875 | 5,216 |
| **Validation** | 8 | 8 | 16 |
| **Test** | 234 | 390 | 624 |

**Note:** The dataset exhibits class imbalance with ~74% pneumonia cases in training data.

### Preprocessing & Augmentation

**Image Processing:**
- Resized to **150×150 pixels**
- Normalized to **[0, 1]** range using rescaling (1/255)
- Grayscale conversion for traditional ML models

**Data Augmentation** (Training only):
```python
- Rotation: ±20°
- Width/Height shift: ±20%
- Shear transformation: 20%
- Zoom: ±20%
- Horizontal flip
```

**Dimensionality Reduction:**
- **PCA** applied for traditional ML models
- Reduced from 22,500 features to **500 components**
- Retained **94.2% variance**

---

## Methods

### Approach

We implemented a **comparative study** of 8 different models across two paradigms:

#### 1. Traditional Machine Learning (5 models)
- Feature extraction via PCA
- Models: Logistic Regression, SVM (RBF kernel), Random Forest, Decision Tree, Naive Bayes

#### 2. Deep Learning (3 models)
- **Custom CNN:** 4 convolutional blocks with batch normalization
- **VGG16 Transfer Learning:** Pre-trained on ImageNet, frozen base layers
- **ResNet50 Transfer Learning:** Pre-trained on ImageNet, frozen base layers

### Why This Approach?

**Rationale:**
1. **Baseline Comparison:** Traditional ML provides interpretable baselines
2. **Transfer Learning:** Leverages pre-trained knowledge from ImageNet to compensate for limited medical imaging data
3. **Custom CNN:** Tests whether domain-specific architecture can outperform generic pre-trained models
4. **Ensemble Potential:** Multiple models enable future ensemble techniques

### Model Architectures

#### Custom CNN Architecture
```
Conv2D(32) → BatchNorm → MaxPool →
Conv2D(64) → BatchNorm → MaxPool →
Conv2D(128) → BatchNorm → MaxPool →
Conv2D(256) → BatchNorm → MaxPool →
Flatten → Dense(512) → Dropout(0.5) →
Dense(256) → Dropout(0.5) → Dense(1, sigmoid)

Total Parameters: 6,944,961 (26.49 MB)
```

#### VGG16 Transfer Learning
```
VGG16 (frozen, 14.7M params) →
GlobalAveragePooling2D →
Dense(256) → Dropout(0.5) →
Dense(128) → Dropout(0.3) →
Dense(1, sigmoid)

Total Parameters: 14,879,041 (56.76 MB)
Trainable: 164,353 | Frozen: 14,714,688
```

#### ResNet50 Transfer Learning
```
ResNet50 (frozen, 23.6M params) →
GlobalAveragePooling2D →
Dense(512) → Dropout(0.5) →
Dense(256) → Dropout(0.3) →
Dense(1, sigmoid)

Total Parameters: 24,768,385 (94.48 MB)
Trainable: 1,180,673 | Frozen: 23,587,712
```

### Training Configuration

**Hyperparameters:**
- **Optimizer:** Adam (learning_rate=0.0001)
- **Loss:** Binary Crossentropy
- **Batch Size:** 32
- **Epochs:** 10 (with early stopping)
- **Metrics:** Accuracy, Precision, Recall

**Callbacks:**
- **EarlyStopping:** patience=5, monitor=val_loss
- **ReduceLROnPlateau:** factor=0.2, patience=3
- **ModelCheckpoint:** Save best model based on val_accuracy

---

## Steps to Run the Code

### Prerequisites
```bash
# Python 3.8+
# TensorFlow 2.x
# Required libraries
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow pillow joblib
```

### Directory Structure
```
pneumonianal-detection/
├── chest_xray/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── models/              # Saved models
├── test_results/        # Evaluation outputs
└── pneumonia_detection.ipynb       # Evaluation outputs
└── app.py
```

### Execution Steps

**Step 1: Dataset Setup**
```python
# Update dataset path in the notebook
DATASET_PATH = 'chest_xray'
```

**Step 2: Run Training Notebook**
```bash
jupyter notebook pneumonia_detection.ipynb
# Execute all cells sequentially
```

**Step 3: Model Training**
The notebook will automatically:
1. Load and preprocess data
2. Train all 8 models
3. Save models in `models/` directory
4. Generate performance visualizations

**Step 4: Test on New Data**
```python
# Configure test dataset path
NEW_TEST_DIR = 'data/chest_xray/test'
MODEL_PATH = 'models/vgg16_best_model.h5'

# Run evaluation section
# Results saved to test_results/
```

### Saved Outputs
- **Models:** `models/*.pkl` (ML), `models/*.h5` (DL)
- **Metrics:** `model_comparison_results.csv`, `performance_metrics.csv`
- **Visualizations:** Confusion matrices, ROC curves, training histories
- **Inference Function:** `predict_pneumonia(image_path, model)`

---

## Experiments & Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **VGG16** ⭐ | **0.8686** | **0.8909** | 0.9000 | **0.8954** |
| Custom CNN | 0.8654 | 0.9091 | 0.8718 | 0.8901 |
| SVM | 0.7708 | 0.7361 | **0.9872** | 0.8434 |
| Logistic Regression | 0.7420 | 0.7116 | 0.9872 | 0.8271 |
| Decision Tree | 0.7420 | 0.7304 | 0.9308 | 0.8185 |
| Naive Bayes | 0.7131 | 0.7221 | 0.8795 | 0.7931 |
| Random Forest | 0.6747 | 0.6587 | 0.9949 | 0.7926 |
| ResNet50 | 0.6250 | 0.6250 | 1.0000 | 0.7692 |

### Best Model: VGG16 Test Performance

**Comprehensive Metrics:**
- **Accuracy:** 85.42%
- **Precision:** 86.73%
- **Recall (Sensitivity):** 90.51%
- **Specificity:** 76.92%
- **F1-Score:** 88.58%
- **ROC AUC:** 92.96%
- **NPV (Negative Predictive Value):** 82.95%

**Confusion Matrix:**
```
                Predicted
              NORMAL  PNEUMONIA
Actual NORMAL    180       54     (76.9% correct)
     PNEUMONIA    37      353     (90.5% correct)
```

**Error Analysis:**
- **False Positive Rate:** 23.08% (54 normal cases misclassified)
- **False Negative Rate:** 9.49% (37 pneumonia cases missed)

### Key Observations

**1. Transfer Learning Superiority**
- VGG16 and Custom CNN significantly outperformed traditional ML (~14-20% accuracy gain)
- Pre-trained features from ImageNet generalize well to medical imaging

**2. Recall vs. Precision Trade-off**
- Traditional ML models (SVM, LR) achieved high recall (>98%) but lower precision
- Deep learning models balanced both metrics better (F1 ~0.89)
- **Clinical Implication:** High recall minimizes missed pneumonia cases

**3. ResNet50 Underperformance**
- Despite deeper architecture, ResNet50 showed poor results (62.5% accuracy)
- Likely causes: Overfitting to ImageNet features, insufficient fine-tuning epochs
- Validation loss increased after epoch 1, suggesting training instability

**4. Model Confidence**
- Average prediction confidence: **83.54%**
- Confidence range: 50.27% - 99.63%
- VGG16 shows well-calibrated probability distributions

### Hyperparameter Experiments

**Learning Rate Reduction:**
- Initial: 1e-4 → Reduced to 2e-5 (epoch 5) → 4e-6 (epoch 10)
- Adaptive learning improved Custom CNN accuracy from 80.37% (epoch 1) to 92.08% (epoch 10)

**Batch Normalization Impact:**
- Custom CNN with BatchNorm showed faster convergence vs. baseline CNN
- Reduced internal covariate shift, enabling higher learning rates

**Dropout Regularization:**
- Dropout(0.5) in dense layers prevented overfitting
- Validation accuracy plateaued at epoch 6 for VGG16, indicating optimal regularization

### Visualizations

The project generates:
1. **Sample X-Ray Images** (NORMAL vs. PNEUMONIA)
   ![Image1](images/sample_images.png)
2. **Model Comparison Bar Charts** (Accuracy, Precision, Recall, F1)
![Image2](images/model_comparison.png)
3. **Training History Plots** (Loss & Accuracy curves)
![Image3](images/custom_cnn_training_history.png)
4. **Confusion Matrices** (All models)
![Image4](images/svm_confusion_matrix.png)
5. **ROC Curve** (AUC = 0.9296)
![Image5](images/roc.png)

6. **Prediction Probability Distributions**
![Image6](images/predicted.png)

---

## Conclusion

### Key Findings

1. **VGG16 Transfer Learning** emerged as the best model with **85.42% test accuracy** and **92.96% ROC AUC**, demonstrating strong generalization to pneumonia detection.

2. **Deep Learning Advantage:** CNN-based models outperformed traditional ML by 14-20% in accuracy, validating the superiority of learned hierarchical features over handcrafted PCA features.

3. **Clinical Viability:** The model achieves **90.51% recall**, meaning it correctly identifies 9 out of 10 pneumonia cases—critical for medical screening where false negatives are costly.

4. **Model Efficiency:** VGG16 with only **164K trainable parameters** (frozen base) achieved comparable performance to custom CNN (6.9M params), demonstrating efficient transfer learning.

5. **Trade-off Analysis:** 
   - **High Recall Models** (SVM, LR): Suitable for screening (minimize missed cases)
   - **Balanced Models** (VGG16, Custom CNN): Suitable for diagnostic confirmation (balance sensitivity & specificity)

### Learned Insights

- **Data Imbalance Handling:** Augmentation helped, but class weights could further improve minority class (NORMAL) performance
- **Transfer Learning Effectiveness:** ImageNet pre-training provides robust low-level visual features applicable to medical imaging
- **Validation Set Limitation:** Only 16 validation images led to noisy validation metrics; larger validation split recommended
- **Ensemble Potential:** Combining VGG16 (high precision) with SVM (high recall) could optimize clinical utility

### Future Improvements

1. **Ensemble Methods:** Combine VGG16 + Custom CNN predictions
2. **Class Weighting:** Address dataset imbalance (74% pneumonia)
3. **Fine-Tuning:** Unfreeze top layers of VGG16 for domain adaptation
4. **Data Augmentation:** Advanced techniques (CutMix, MixUp)
5. **Explainability:** Integrate Grad-CAM for visualizing decision regions
6. **Multi-Class Extension:** Classify pneumonia subtypes (viral/bacterial)

---

## References

1. **Dataset:** Kermany, D. et al. (2018). "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification." Mendeley Data, v2.
2. **VGG16:** Simonyan, K. & Zisserman, A. (2015). "Very Deep Convolutional Networks for Large-Scale Image Recognition." ICLR 2015.
3. **ResNet50:** He, K. et al. (2016). "Deep Residual Learning for Image Recognition." CVPR 2016.
4. **Transfer Learning:** Yosinski, J. et al. (2014). "How transferable are features in deep neural networks?" NIPS 2014.
5. **Medical Imaging:** Rajpurkar, P. et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning." arXiv:1711.05225.
6. **TensorFlow Documentation:** https://www.tensorflow.org/api_docs
7. **Scikit-learn Documentation:** https://scikit-learn.org/stable/documentation.html

---
