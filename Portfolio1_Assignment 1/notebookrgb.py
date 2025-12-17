"""
Waterfowl Detection in UAV Thermal + RGB Imagery using YOLOv8
--------------------------------------------------------------
Assignment 1: Object Detection for Wildlife Conservation

This script implements a complete pipeline for detecting waterfowl using both
thermal and RGB imagery:
1. Dataset preparation with both modalities
2. Model training with YOLOv8
3. Evaluation on test set
4. Visualization of results

Author: [Your Team Name]
Date: November 2025
"""

import os
import random
import shutil
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch
import pandas as pd

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class MultiModalWaterfowlDetection:
    """
    Complete pipeline for waterfowl detection using thermal and/or RGB imagery.
    """
    
    def __init__(self, data_root, output_root='output', use_rgb=True, use_thermal=True):
        """
        Initialize the pipeline.
        
        Args:
            data_root: Path to the dataset root directory
            output_root: Path to save all outputs
            use_rgb: Include RGB images in training
            use_thermal: Include thermal images in training
        """
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(exist_ok=True)
        
        self.use_rgb = use_rgb
        self.use_thermal = use_thermal
        
        if not use_rgb and not use_thermal:
            raise ValueError("At least one modality (RGB or thermal) must be enabled!")
        
        # Create directory structure for YOLO format
        self.yolo_root = self.output_root / 'yolo_dataset'
        self.model_dir = self.output_root / 'models'
        self.results_dir = self.output_root / 'results'
        
        print("\n✓ Cleaning all old output directories...")
        # Clean and create output directories
        for d in [self.yolo_root, self.model_dir, self.results_dir]:
            if d.exists():
                print(f"  Removing {d}")
                shutil.rmtree(d)  # Remove all old data
            d.mkdir(parents=True, exist_ok=True)  # Create fresh directories
        
        modalities = []
        if use_thermal:
            modalities.append("Thermal")
        if use_rgb:
            modalities.append("RGB")
        
        print(f"✓ Pipeline initialized with: {' + '.join(modalities)}")
        print(f"  Data root: {self.data_root}")
        print(f"  Output root: {self.output_root}")
    
    def prepare_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Prepare and split the dataset into train/val/test sets.
        Includes both thermal and RGB images.
        Reads CSV annotations and converts to YOLO format.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
        """
        print("\n" + "="*60)
        print("STEP 1: Dataset Preparation (Multi-Modal)")
        print("="*60)
        
        # Define paths - RGB folder is at same level as thermal_Images
        thermal_img_dir = self.data_root / 'thermal_Images' / '01_Positive_img'
        thermal_annotation_dir = self.data_root / 'thermal_Images' / '02_Positive_annotation'
        thermal_csv_path = thermal_annotation_dir / 'Bounding Box Label.csv'
        
        rgb_img_dir = self.data_root / 'RGB_Images' / '01_Positive_img'
        rgb_annotation_dir = self.data_root / 'RGB_Images' / '02_Positive_annotation'
        rgb_csv_path = rgb_annotation_dir / 'Bounding Box Label.csv'
        
        # Check if paths exist
        if self.use_thermal:
            if not thermal_img_dir.exists():
                raise FileNotFoundError(f"Thermal image directory not found: {thermal_img_dir}")
            if not thermal_csv_path.exists():
                raise FileNotFoundError(f"Thermal CSV file not found: {thermal_csv_path}")
        
        if self.use_rgb:
            if not rgb_img_dir.exists():
                raise FileNotFoundError(f"RGB image directory not found: {rgb_img_dir}")
            if not rgb_csv_path.exists():
                raise FileNotFoundError(f"RGB CSV file not found: {rgb_csv_path}")
        
        # Read CSV annotations for both modalities
        print(f"\n✓ Reading annotations from CSV files...")
        
        # Create separate annotation dictionaries for thermal and RGB
        thermal_annotations_dict = {}
        rgb_annotations_dict = {}
        
        if self.use_thermal:
            df_thermal = pd.read_csv(thermal_csv_path)
            print(f"✓ Found {len(df_thermal)} thermal annotations in CSV")
            print(f"✓ Thermal CSV columns: {list(df_thermal.columns)}")
            
            for _, row in df_thermal.iterrows():
                img_name = row['imageFilename']
                if img_name not in thermal_annotations_dict:
                    thermal_annotations_dict[img_name] = []
                thermal_annotations_dict[img_name].append(row)
            
            print(f"✓ Found annotations for {len(thermal_annotations_dict)} unique thermal images")
        
        if self.use_rgb:
            df_rgb = pd.read_csv(rgb_csv_path)
            print(f"✓ Found {len(df_rgb)} RGB annotations in CSV")
            print(f"✓ RGB CSV columns: {list(df_rgb.columns)}")
            
            # Create mapping: CSV filename (*.tif) -> actual RGB filename (*.jpg with _visual suffix)
            for _, row in df_rgb.iterrows():
                csv_filename = row['imageFilename']  # e.g., "20180322_101536_979_R.tif"
                
                # Convert .tif to _visual.jpg format
                # Remove .tif extension and add _visual.jpg
                base_name = csv_filename.replace('.tif', '').replace('.TIF', '')
                rgb_filename = f"{base_name}_visual.jpg"
                
                if rgb_filename not in rgb_annotations_dict:
                    rgb_annotations_dict[rgb_filename] = []
                rgb_annotations_dict[rgb_filename].append(row)
            
            print(f"✓ Found annotations for {len(rgb_annotations_dict)} unique RGB images (mapped from CSV)")
            
            # Show sample mappings for verification
            if len(rgb_annotations_dict) > 0:
                print(f"\n✓ Sample RGB filename mappings (CSV -> Actual):")
                for i, (actual_name, _) in enumerate(list(rgb_annotations_dict.items())[:3]):
                    # Reverse engineer the CSV name
                    csv_name = actual_name.replace('_visual.jpg', '.tif')
                    print(f"  {csv_name} → {actual_name}")
        
        # Collect images from both modalities
        all_images = []
        
        if self.use_thermal:
            thermal_files = list(thermal_img_dir.glob('*.tif'))
            valid_thermal = [(f, 'thermal') for f in thermal_files if f.name in thermal_annotations_dict]
            all_images.extend(valid_thermal)
            print(f"\n✓ Found {len(valid_thermal)} thermal images with annotations")
            
            if len(valid_thermal) < len(thermal_files):
                print(f"  ⚠ Note: {len(thermal_files) - len(valid_thermal)} thermal images without annotations (skipped)")
        
        if self.use_rgb:
            # RGB images - look for common image formats
            rgb_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                rgb_files.extend(list(rgb_img_dir.glob(ext)))
            
            print(f"\n✓ Found {len(rgb_files)} RGB images in directory")
            
            # Match RGB images to their own annotations
            valid_rgb = [(f, 'rgb') for f in rgb_files if f.name in rgb_annotations_dict]
            
            all_images.extend(valid_rgb)
            print(f"✓ Found {len(valid_rgb)} RGB images with annotations")
            
            if len(valid_rgb) < len(rgb_files):
                print(f"  ⚠ Note: {len(rgb_files) - len(valid_rgb)} RGB images without annotations (skipped)")
        
        if len(all_images) == 0:
            print("\n⚠ WARNING: No matching images found!")
            if self.use_thermal:
                print("Sample thermal annotation filenames:")
                for name in list(thermal_annotations_dict.keys())[:5]:
                    print(f"  - {name}")
                print("\nSample thermal image filenames:")
                for img in list(thermal_img_dir.glob('*.tif'))[:5]:
                    print(f"  - {img.name}")
            if self.use_rgb:
                print("\nSample RGB annotation filenames (mapped):")
                for name in list(rgb_annotations_dict.keys())[:5]:
                    print(f"  - {name}")
                print("\nSample RGB image filenames:")
                rgb_files_sample = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
                    rgb_files_sample.extend(list(rgb_img_dir.glob(ext)))
                for img in rgb_files_sample[:5]:
                    print(f"  - {img.name}")
            raise ValueError("No images match the annotations in the CSV files")
        
        print(f"\n✓ Total images for training: {len(all_images)}")
        if self.use_thermal and self.use_rgb:
            thermal_count = sum(1 for _, mod in all_images if mod == 'thermal')
            rgb_count = sum(1 for _, mod in all_images if mod == 'rgb')
            print(f"  - Thermal: {thermal_count}")
            print(f"  - RGB: {rgb_count}")
        
        # Split dataset
        train_imgs, temp_imgs = train_test_split(
            all_images, 
            test_size=(val_ratio + test_ratio),
            random_state=42
        )
        
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=42
        )
        
        print(f"\n✓ Dataset split:")
        print(f"  Training:   {len(train_imgs)} images ({train_ratio*100:.0f}%)")
        print(f"  Validation: {len(val_imgs)} images ({val_ratio*100:.0f}%)")
        print(f"  Test:       {len(test_imgs)} images ({test_ratio*100:.0f}%)")
        
        # Create YOLO directory structure
        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        # Clear old dataset directories
        print("\n✓ Clearing old dataset directories...")
        for split_name in splits:
            split_dir = self.yolo_root / split_name
            if split_dir.exists():
                print(f"  Removing {split_dir}")
                shutil.rmtree(split_dir)
        
        for split_name, img_list in splits.items():
            # Create directories
            img_dir = self.yolo_root / split_name / 'images'
            label_dir = self.yolo_root / split_name / 'labels'
            img_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n✓ Processing {split_name} set...")
            processed_count = 0
            skipped_count = 0
            
            for img_path, modality in img_list:
                # Read image based on modality
                if modality == 'thermal':
                    # Read thermal as grayscale and convert to RGB
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    # Use thermal annotations
                    annotations = thermal_annotations_dict.get(img_path.name, [])
                else:  # RGB
                    # Read RGB directly
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Use RGB annotations
                    annotations = rgb_annotations_dict.get(img_path.name, [])
                
                if img is None:
                    print(f"  ⚠ Warning: Could not read {img_path.name}")
                    skipped_count += 1
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # Create unique filename (prefix with modality to avoid naming conflicts)
                dest_filename = f"{modality}_{img_path.stem}.jpg"
                dest_img = img_dir / dest_filename
                
                # Save image as JPG for consistency
                cv2.imwrite(str(dest_img), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
                # Check if annotations exist
                if len(annotations) == 0:
                    print(f"  ⚠ Warning: No annotations found for {img_path.name}")
                    if dest_img.exists():
                        dest_img.unlink()
                    skipped_count += 1
                    continue
                
                # Convert annotations to YOLO format
                yolo_annotations = []
                invalid_count = 0
                
                for ann in annotations:
                    # Get bounding box in pixel coordinates from CSV
                    x1_px = float(ann['x(column)'])
                    y1_px = float(ann['y(row)'])
                    w_px = float(ann['width'])
                    h_px = float(ann['height'])
                    
                    # Skip boxes with zero or negative dimensions
                    if w_px <= 0 or h_px <= 0:
                        invalid_count += 1
                        continue
                    
                    # Calculate x2, y2 in pixels
                    x2_px = x1_px + w_px
                    y2_px = y1_px + h_px
                    
                    # Clip pixel coordinates to image boundaries
                    x1_clipped = max(0.0, x1_px)
                    y1_clipped = max(0.0, y1_px)
                    x2_clipped = min(float(img_width), x2_px)
                    y2_clipped = min(float(img_height), y2_px)
                    
                    # Recalculate width/height from the clipped coordinates
                    w_clipped = x2_clipped - x1_clipped
                    h_clipped = y2_clipped - y1_clipped
                    
                    # Skip if box is now invalid
                    if w_clipped <= 0.001 or h_clipped <= 0.001:
                        invalid_count += 1
                        continue
                    
                    # Convert to YOLO format (normalized coordinates)
                    x_center = (x1_clipped + w_clipped / 2) / img_width
                    y_center = (y1_clipped + h_clipped / 2) / img_height
                    norm_width = w_clipped / img_width
                    norm_height = h_clipped / img_height
                    
                    # Class 0 for waterfowl
                    yolo_annotations.append(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
                
                if invalid_count > 0 and split_name == 'train':
                    print(f"  ⚠ {img_path.name} ({modality}): filtered {invalid_count} invalid boxes")
                
                # Save annotations
                if len(yolo_annotations) > 0:
                    dest_ann = label_dir / f"{modality}_{img_path.stem}.txt"
                    with open(dest_ann, 'w') as f:
                        f.write('\n'.join(yolo_annotations))
                    processed_count += 1
                else:
                    # Remove image if no valid annotations
                    if dest_img.exists():
                        dest_img.unlink()
                    skipped_count += 1
                    if split_name == 'train':
                        print(f"  ⚠ {img_path.name} ({modality}): skipped (no valid annotations)")
            
            print(f"  Processed {processed_count} images successfully")
            if skipped_count > 0:
                print(f"  Skipped {skipped_count} images (no valid annotations or read errors)")
        
        # Create data.yaml for YOLO
        yaml_content = f"""# Multi-Modal Waterfowl Detection Dataset
path: {self.yolo_root.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: 1  # number of classes
names: ['waterfowl']  # class names

# Dataset info
# Modalities: {'Thermal + RGB' if (self.use_thermal and self.use_rgb) else 'Thermal only' if self.use_thermal else 'RGB only'}
"""
        
        yaml_path = self.yolo_root / 'data.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✓ Dataset prepared successfully!")
        print(f"  YOLO format saved to: {self.yolo_root}")
        print(f"  Configuration: {yaml_path}")
        
        # Validate the dataset
        self._validate_dataset()
        
        return yaml_path
    
    def _validate_dataset(self):
        """
        Validate that all labels are properly formatted AND match images.
        """
        print(f"\n✓ Validating dataset...")
        
        overall_issues = 0
        
        for split in ['train', 'val', 'test']:
            img_dir = self.yolo_root / split / 'images'
            label_dir = self.yolo_root / split / 'labels'
            
            if not img_dir.exists() or not label_dir.exists():
                print(f"  Skipping {split}: directory not found.")
                continue

            # Get stems of all image and label files
            img_files = {p.stem for p in img_dir.glob('*.*') if p.suffix in ['.jpg', '.jpeg', '.png', '.tif']}
            label_files = {p.stem for p in label_dir.glob('*.txt')}

            # Check for orphan labels
            orphan_labels = label_files - img_files
            if orphan_labels:
                print(f"  ⚠ {split}: Found {len(orphan_labels)} orphan label files (no matching image).")
                for orphan in list(orphan_labels)[:3]:
                    print(f"    - {orphan}.txt")
                overall_issues += len(orphan_labels)

            # Check for images without labels
            background_images = img_files - label_files
            if background_images:
                print(f"  ℹ {split}: Found {len(background_images)} images with no labels (treated as backgrounds).")
            
            # Check all existing label files
            issues_in_split = 0
            total_boxes = 0
            empty_files = 0
            
            for label_stem in (label_files & img_files):
                label_file = label_dir / f"{label_stem}.txt"
                
                # Check for empty files
                if label_file.stat().st_size == 0:
                    print(f"  ⚠ {label_file.name}: FILE IS EMPTY (0 bytes).")
                    issues_in_split += 1
                    empty_files += 1
                    continue
                    
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        print(f"  ⚠ {label_file.name}: FILE IS EMPTY (no lines).")
                        issues_in_split += 1
                        empty_files += 1
                        continue

                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split()
                        if len(parts) != 5:
                            print(f"  ⚠ {label_file.name} line {line_num}: wrong format (expected 5 parts, got {len(parts)})")
                            issues_in_split += 1
                            continue
                        
                        try:
                            cls, x, y, w, h = map(float, parts)
                            total_boxes += 1
                            
                            # Check ranges
                            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                                print(f"  ⚠ {label_file.name} line {line_num}: values out of range")
                                print(f"    x={x:.3f}, y={y:.3f}, w={w:.3f}, h={h:.3f}")
                                issues_in_split += 1
                        except ValueError:
                            print(f"  ⚠ {label_file.name} line {line_num}: invalid values (not numbers)")
                            issues_in_split += 1
            
            print(f"  {split}: {len(label_files)} labels, {len(img_files)} images.")
            print(f"    - {total_boxes} total boxes.")
            if empty_files > 0:
                print(f"    - {empty_files} empty label files.")
            if issues_in_split - empty_files > 0:
                print(f"    - {issues_in_split - empty_files} line/format issues.")
            overall_issues += issues_in_split
        
        if overall_issues > 0:
            print(f"\n⚠ WARNING: Found {overall_issues} total annotation issues.")
        else:
            print(f"\n✓ All annotations are valid!")
    
    def train_model(self, yaml_path, epochs=100, img_size=640, batch_size=16):
        """
        Train YOLOv8 model on the prepared dataset.
        
        Args:
            yaml_path: Path to data.yaml
            epochs: Number of training epochs
            img_size: Input image size
            batch_size: Batch size for training
        """
        print("\n" + "="*60)
        print("STEP 2: Model Training")
        print("="*60)
        
        device = 'cpu'
        
        # Load pretrained YOLOv8 model
        print("\n✓ Loading YOLOv8n (nano) pretrained model...")
        model = YOLO('yolov8n.pt')
        
        print(f"\n✓ Starting training...")
        print(f"  Epochs: {epochs}")
        print(f"  Image size: {img_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {device}")
        
        # Train the model
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            project=str(self.model_dir),
            name='multimodal_detector',
            exist_ok=True,
            # Data augmentation settings
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.0,
        )
        
        # Save the best model path
        self.best_model_path = self.model_dir / 'multimodal_detector' / 'weights' / 'best.pt'
        
        print(f"\n✓ Training complete!")
        print(f"  Best model saved to: {self.best_model_path}")
        
        return results
    
    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        """
        print("\n" + "="*60)
        print("STEP 3: Model Evaluation")
        print("="*60)
        
        # Load the best model
        model = YOLO(str(self.best_model_path))
        
        device = 'cpu'
        
        # Run validation on test set
        print("\n✓ Evaluating on test set...")
        yaml_path = self.yolo_root / 'data.yaml'
        
        # Validate on test split
        metrics = model.val(
            data=str(yaml_path),
            split='test',
            device=device
        )
        
        print("\n✓ Evaluation Results:")
        print(f"  mAP@50:    {metrics.box.map50:.4f}")
        print(f"  mAP@50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall:    {metrics.box.mr:.4f}")
        
        return metrics
    
    def visualize_results(self, num_examples=3):
        """
        Visualize detection results: true positives, false positives, false negatives.
        
        Args:
            num_examples: Number of examples to visualize for each category
        """
        print("\n" + "="*60)
        print("STEP 4: Visualization")
        print("="*60)
        
        model = YOLO(str(self.best_model_path))
        device = 'cpu'
        
        # Get test images
        test_img_dir = self.yolo_root / 'test' / 'images'
        test_label_dir = self.yolo_root / 'test' / 'labels'
        test_images = list(test_img_dir.glob('*.jpg')) + list(test_img_dir.glob('*.png'))
        
        # Categories for visualization
        categories = {
            'true_positives': [],
            'false_negatives': [],
            'false_positives': []
        }
        
        print(f"\n✓ Analyzing {len(test_images)} test images...")
        
        # Run inference and categorize results
        for img_path in test_images:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get predictions
            results = model(img_path, device=device, verbose=False)
            predictions = results[0].boxes
            
            # Load ground truth
            label_path = test_label_dir / f"{img_path.stem}.txt"
            gt_boxes = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            gt_boxes.append([float(x) for x in parts[1:]])
            
            # Categorize
            num_predictions = len(predictions)
            num_ground_truth = len(gt_boxes)
            
            if num_ground_truth > 0 and num_predictions > 0:
                categories['true_positives'].append((img_path, img_rgb, predictions, gt_boxes))
            elif num_ground_truth > 0 and num_predictions == 0:
                categories['false_negatives'].append((img_path, img_rgb, predictions, gt_boxes))
            elif num_ground_truth == 0 and num_predictions > 0:
                categories['false_positives'].append((img_path, img_rgb, predictions, gt_boxes))
        
        # Visualize examples
        for category_name, examples in categories.items():
            if len(examples) == 0:
                print(f"\n! No {category_name} found")
                continue
            
            # Select random examples
            selected = random.sample(examples, min(num_examples, len(examples)))
            
            print(f"\n✓ Visualizing {category_name}: {len(selected)} examples")
            
            fig, axes = plt.subplots(1, len(selected), figsize=(5*len(selected), 5))
            if len(selected) == 1:
                axes = [axes]
            
            for idx, (img_path, img_rgb, predictions, gt_boxes) in enumerate(selected):
                ax = axes[idx]
                ax.imshow(img_rgb)
                
                h, w = img_rgb.shape[:2]
                
                # Draw ground truth boxes (green)
                for box in gt_boxes:
                    x_center, y_center, width, height = box
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                        fill=False, color='green', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, 'GT', color='green', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Draw predictions (red)
                if len(predictions) > 0:
                    for box in predictions.xyxy:
                        x1, y1, x2, y2 = box.cpu().numpy()
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                            fill=False, color='red', linewidth=2)
                        ax.add_patch(rect)
                        ax.text(x1, y1-5, 'Pred', color='red', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                ax.set_title(f"{img_path.name}", fontsize=10)
                ax.axis('off')
            
            plt.tight_layout()
            save_path = self.results_dir / f'{category_name}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Saved to: {save_path}")
            plt.close()
        
        print(f"\n✓ Visualization complete!")


def main():
    """
    Main execution function.
    """
    print("\n" + "="*60)
    print("MULTI-MODAL WATERFOWL DETECTION PIPELINE")
    print("="*60)
    
    # Configuration
    DATA_ROOT = '/Users/harshasathish/Desktop/MAI/ComputerVision/Portfolio1_new'
    OUTPUT_ROOT = 'output'
    
    # ============================================================
    # CONFIGURE WHICH MODALITIES TO USE
    # ============================================================
    USE_THERMAL = True   # Include thermal images
    USE_RGB = True       # Include RGB images
    # Set both to True to train on both modalities together
    # Set only one to True to train on single modality
    # ============================================================
    
    # Training hyperparameters
    EPOCHS = 100
    IMG_SIZE = 640
    BATCH_SIZE = 8
    
    # Initialize pipeline
    pipeline = MultiModalWaterfowlDetection(
        data_root=DATA_ROOT,
        output_root=OUTPUT_ROOT,
        use_thermal=USE_THERMAL,
        use_rgb=USE_RGB
    )
    
    # Step 1: Prepare dataset
    yaml_path = pipeline.prepare_dataset(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Step 2: Train model
    pipeline.train_model(
        yaml_path=yaml_path,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Step 3: Evaluate model
    pipeline.evaluate_model()
    
    # Step 4: Visualize results
    pipeline.visualize_results(num_examples=3)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_ROOT}/")
    
    modalities_used = []
    if USE_THERMAL:
        modalities_used.append("Thermal")
    if USE_RGB:
        modalities_used.append("RGB")
    print(f"Modalities used: {' + '.join(modalities_used)}")
    print("\nNext steps:")
    print("  1. Check training results in: output/models/multimodal_detector/")
    print("  2. View visualizations in: output/results/")
    print("  3. Compare performance with thermal-only or RGB-only models")


if __name__ == '__main__':
    main()