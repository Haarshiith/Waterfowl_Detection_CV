# Computer Vision - Portfolio Exam 1

# Object Detection for Wildlife Conservation - Detecting Waterfowl in UAV Thermal Imagery

**Submitted by:**

* Riya Biju - 10000742
* Harsha Sathish - 10001000
* Harshith Babu Prakash Babu - 10001191

---

**Project Overview**

This project implements an automated waterfowl detection system using YOLOv8 on UAV-captured thermal imagery. The system enables real-time detection and counting of waterfowl in wetland environments for wildlife conservation monitoring.

**Key Features:**
* Real-time waterfowl detection using YOLOv8 Nano
* Thermal imagery processing with CLAHE contrast enhancement
* Optimized for small object detection (5-15 pixels)
* Achieves 93.75% mAP@50 on test set
* Comprehensive error analysis with TP, FP, FN visualization

---

**Prerequisites**

Download the dataset files - both Thermal and RGB Images from NetInstall the required Python libraries:

```bash
pip install ultralytics opencv-python matplotlib pandas scikit-learn torch
```

---

**Repository Structure**

```
├── README.md                                # Top-level README
│
├── Portfolio1_Assignment 1/                # Main assignment folder
│   ├── Portfolio1_CV.ipynb                   # Main notebook - Best model (Thermal only, no mosaic)
│   ├── notebookrgb.py                        # RGB fusion experiment (Thermal + RGB)
│   │
│   ├── output_final/                         # Outputs from best model
│   │   ├── models/                            # Trained YOLOv8 weights
│   │   │   └── waterfowl_detector/
│   │   │       └── weights/
│   │   │           ├── best.pt                # Best checkpoint
│   │   │           └── last.pt                # Last epoch
│   │   ├── results/                           # Error analysis visualizations
│   │   │   ├── true_positives.png            # Correct detections
│   │   │   ├── false_positives.png           # Incorrect detections
│   │   │   └── false_negatives.png           # Missed detections
│   │   └── yolo_dataset/                      # YOLO format dataset
│   │       └── data.yaml                      # Dataset config
│   │
│   └── experiment_runs/                       # RGB fusion experiment outputs
│
├── Portfolio2_Assignment 2/                 # Second assignment folder
│   └── Portfolio2 - Assignment 2.pdf          # Assignment 2 PDF
│
└── Presentations/                             # Presentation folder
    └── Portfolio1_Assignment1_Presentation1/2.pdf             # Canva presentation PDF
```

---

**Key Components**

1. **Data Preprocessing Pipeline**
   * Converts single-channel thermal to 3-channel (channel replication)
   * Resizes images to 640×640 with padding
   * Converts CSV annotations to YOLO format
   * Creates empty labels for negative images

2. **YOLOv8 Nano Model**
   * Single-stage object detector optimized for speed
   * Pre-trained on COCO dataset, fine-tuned for waterfowl
3. **Training Configuration**
   * 100 epochs
   * Batch size: 8, Image size: 640×640
   * Augmentations: flips, rotation, brightness, contrast
   * **Mosaic augmentation disabled

4. **Evaluation & Visualization**
   * Metrics: mAP@50, mAP@50-95, Precision, Recall
   * Error analysis: True Positives, False Positives, False Negatives
   * Visual comparison of detections vs ground truth

---

**Usage**

**Running the Best Model (Thermal Only):**

```python
# Open and run Portfolio1_CV.ipynb in Jupyter Notebook or Google Colab
# The notebook will:
# 1. Prepare dataset with train/val/test split
# 2. Train YOLOv8 model on thermal images
# 3. Evaluate on test set
# 4. Generate visualization of TP, FP, FN
# 5. Save trained model to output/models/
```

**Running RGB Fusion Experiment:**

```python
# Run the RGB fusion script
python notebookrgb.py

# This experiment combines thermal and RGB images
# Results saved to runs/ directory
```

---

**Acknowledgments**

Thanks to Prof. Dr. Dominik Seuß for guidance on Computer Vision.

Dataset: UAV-derived Waterfowl Thermal Imagery Dataset (Mendeley Data)
