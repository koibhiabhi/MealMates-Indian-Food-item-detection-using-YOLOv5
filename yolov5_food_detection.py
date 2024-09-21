# Import necessary libraries
import os
import glob
import matplotlib.pyplot as plt
import cv2
import requests
import random
import numpy as np
import datetime
import subprocess
import pandas as pd
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)

# Flag to indicate whether to train the model or not
TRAIN = True

# Number of epochs to train for
EPOCHS = 50

# Dataset structure:
# dataset_f/
# ├── data.yaml
# ├── train/
# │   ├── images/
# │   ├── labels/
# ├── val/
# │   ├── images/
# │   ├── labels/
# └── test/
#     ├── images/
#     ├── labels/

# Path to dataset folder
dataset_path = 'C:/Users/avina/dataset_f/'

# List of directories to process
dirs = ['train', 'valid', 'test']

# Iterate over train, val, and test directories
for dir_name in dirs:
    image_dir = os.path.join(dataset_path, dir_name, 'images')
    label_dir = os.path.join(dataset_path, dir_name, 'labels')

    # Ensure the directories exist
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"Directory {image_dir} or {label_dir} does not exist.")
        continue

    # Getting sorted list of all image names
    all_image_names = sorted(os.listdir(image_dir))

# Function to download a file from a URL
def download_file(url, save_name):
    if not os.path.exists(save_name):
        file = requests.get(url)
        open(save_name, 'wb').write(file.content)
    else: 
        print('File already present, skipping download...')

# List of class names for the food items
class_names = ['Aloo', 'Bhindi', 'Dhokla', 'Jalebi', 'Rice', 'Dal', 'Gulab Jamun', 'Idli', 'Kofta', 'Chapati']

# Generate random colors for each class
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# Function to convert YOLO format bounding boxes to (xmin, ymin, xmax, ymax) format
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0] - bboxes[2] / 2, bboxes[1] - bboxes[3] / 2
    xmax, ymax = bboxes[0] + bboxes[2] / 2, bboxes[1] + bboxes[3] / 2
    return xmin, ymin, xmax, ymax

# Function to plot bounding boxes on an image
def plot_box(image, bboxes, labels):
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        xmin, ymin = int(x1*w), int(y1*h)
        xmax, ymax = int(x2*w), int(y2*h)
        
        class_name = class_names[int(labels[box_num])]
        color = colors[class_names.index(class_name)]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color, thickness=2)

        # Adjust font scale and thickness based on image size
        font_scale = min(1, max(3, int(w/500)))
        font_thickness = min(2, max(10, int(w/50)))

        # Add label to the bounding box
        label_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        cv2.rectangle(image, (xmin, ymin-label_size[1]-10), (xmin+label_size[0], ymin), color, -1)
        cv2.putText(image, class_name, (xmin+1, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), font_thickness)

    return image

# Function to plot sample images with bounding boxes
def plot(image_paths, label_paths, num_samples):
    all_training_images = glob.glob(image_paths)
    all_training_labels = glob.glob(label_paths)
    all_training_images.sort()
    all_training_labels.sort()
    
    num_images = len(all_training_images)
    
    plt.figure(figsize=(15, 12))
    for i in range(num_samples):
        j = random.randint(0, num_images-1)
        image = cv2.imread(all_training_images[j])
        with open(all_training_labels[j], 'r') as f:
            bboxes = []
            labels = []
            for line in f.readlines():
                label, *bbox = line.strip().split()
                bboxes.append(list(map(float, bbox)))
                labels.append(label)
        
        result_image = plot_box(image, bboxes, labels)
        plt.subplot(2, 2, i+1)
        plt.imshow(result_image[:, :, ::-1])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize a few training images
plot(
    image_paths='C:/Users/avina/dataset_f/train/images/*',
    label_paths='C:/Users/avina/dataset_f/train/labels/*',
    num_samples=4,
)

# Set environment variable to avoid OpenMP errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Training configuration
data_yaml_path = "C:/Users/avina/dataset_f/data.yaml"
result_dir = "custom_yolov5_training"

# Print timestamp before starting the training process
start_time = datetime.datetime.now()
print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Start YOLOv5 training process (CPU only, no CUDA)
process = subprocess.Popen([
    'python', 'C:/Users/avina/yolov5/train.py',
    '--data', data_yaml_path,
    '--weights', 'yolov5n.pt',  
    '--img', '416',  
    '--epochs', str(EPOCHS),
    '--batch-size', '32',  
    '--workers', '2',  
    '--name', result_dir  
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

# Variables to track time
epoch_start_time = None
elapsed_epochs = 0

# Monitor training progress
for line in process.stdout:
    print(line, end='')
    if "Epoch" in line:
        if epoch_start_time is None:
            epoch_start_time = datetime.datetime.now()
        else:
            current_time = datetime.datetime.now()
            elapsed_time = (current_time - epoch_start_time).total_seconds()
            elapsed_epochs += 1

            # Estimate remaining time
            if elapsed_epochs > 0:
                average_epoch_time = elapsed_time / elapsed_epochs
                remaining_epochs = EPOCHS - elapsed_epochs
                estimated_remaining_time = average_epoch_time * remaining_epochs
                remaining_minutes, remaining_seconds = divmod(int(estimated_remaining_time), 60)
                print(f"Estimated time left: {remaining_minutes}m {remaining_seconds}s")

            epoch_start_time = current_time  # Reset for the next epoch

# Wait for the process to complete
process.wait()

# Print timestamp after the training process completes
end_time = datetime.datetime.now()
print(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Path to the trained model weights
weights_path = 'C:/Users/avina/yolov5/runs/train/custom_yolov5_training4/weights/best.pt'

# Path to the images you want to test
source_path = 'C:/Users/avina/dataset_f/test/images'

# Directory to save inference results
result_dir = 'yolov5_inference_results'

# Run the detection script
process = subprocess.Popen([
    'python', 'C:/Users/avina/yolov5/detect.py',
    '--weights', weights_path,
    '--img', '416',
    '--conf', '0.4',
    '--source', source_path,
    '--name', result_dir
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

# Print output from the inference script in real-time
for line in process.stdout:
    print(line, end='')

# Wait for the inference process to complete
process.wait()

print(f"Inference completed! Check the {result_dir} directory for the results.")

# Directory where the images are saved after inference
results_dir = 'yolov5/runs/detect/yolov5_inference_results13'

# Get list of image files from the directory
image_files = [f for f in os.listdir(results_dir) if f.endswith('.jpg')]

# Select 4 random images from the directory
sample_images = random.sample(image_files, 4)

# Plot the images
plt.figure(figsize=(10, 10))
for i, image_file in enumerate(sample_images):
    image_path = os.path.join(results_dir, image_file)
    img = Image.open(image_path)
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.axis('off')
plt.show()

# Run inference on a specific image
source_image_path = 'C:/Users/avina/Downloads/thh.png'
result_dir = 'yolov5_inference_results'

process = subprocess.Popen([
    'python', 'C:/Users/avina/yolov5/detect.py',
    '--weights', weights_path,
    '--img', '416',
    '--conf', '0.5',
    '--source', source_image_path,
    '--name', result_dir,
    '--device', 'cpu'
], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

# Print output from the inference script in real-time with timestamps
for line in process.stdout:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {line.strip()}")

# Wait for the inference process to complete
process.wait()

# Print completion message with timestamp
end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(f"[{end_time}] Inference completed! Check the {result_dir} directory for the results.")

# Load and analyze training results
results_path = 'C:/Users/avina/yolov5/runs/train/custom_yolov5_training4/results.csv'
data = pd.read_csv(results_path)

# Strip whitespace from column names
data.columns = data.columns.str.strip()

# Extract the relevant columns
epochs = data['epoch']
train_box_loss = data['train/box_loss']
val_box_loss = data['val/box_loss']
train_obj_loss = data['train/obj_loss']
val_obj_loss = data['val/obj_loss']
train_cls_loss = data['train/cls_loss']
val_cls_loss = data['val/cls_loss']
train_precision = data['metrics/precision']
val_recall = data['metrics/recall']
val_mAP = data['metrics/mAP_0.5']

# Create a summary DataFrame
summary_data = pd.DataFrame({
    'Epoch': epochs,
    'Train Box Loss': train_box_loss,
    'Validation Box Loss': val_box_loss,
    'Train Object Loss': train_obj_loss,
    'Validation Object Loss': val_obj_loss,
    'Train Class Loss': train_cls_loss,
    'Validation Class Loss': val_cls_loss,
    'Train Precision': train_precision,
    'Validation Recall': val_recall,
    'Validation mAP': val_mAP
})

# Print the summary table
print(summary_data)

# Save the summary table to a CSV file
summary_data.to_csv('summary_results.csv', index=False)

# Plot Loss and mAP
plt.figure(figsize=(12, 5))

# Plot Loss Curves
plt.subplot(1, 2, 1)
plt.plot(epochs, train_box_loss, label='Train Box Loss', marker='o')
plt.plot(epochs, val_box_loss, label='Validation Box Loss', marker='o')
plt.plot(epochs, train_obj_loss, label='Train Object Loss', marker='o')
plt.plot(epochs, val_obj_loss, label='Validation Object Loss', marker='o')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Plot Mean Average Precision
plt.subplot(1, 2, 2)
plt.plot(epochs, val_mAP, label='Validation mAP', marker='o')
plt.title('Mean Average Precision')
plt.xlabel('Epochs')
plt.ylabel('mAP')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Plot Precision and Recall
plt.figure(figsize=(12, 6))
plt.plot(epochs, train_precision, label='Train Precision', marker='o')
plt.plot(epochs, val_recall, label='Validation Recall', marker='o')
plt.title('Precision and Recall over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()

# Calculate F1-Score
f1_score = 2 * (train_precision * val_recall) / (train_precision + val_recall)

# Create a summary DataFrame including F1-Score
summary_data_with_f1 = pd.DataFrame({
    'Epoch': epochs,
    'Train Precision': train_precision,
    'Validation Recall': val_recall,
    'Validation mAP': val_mAP,
    'F1-Score': f1_score
})

# Print the summary table with F1-Score
print(summary_data_with_f1)

# Save the summary table with F1-Score to a CSV file
summary_data_with_f1.to_csv('summary_results_with_f1.csv', index=False)

# Plot Precision, Recall, and F1-Score
plt.figure(figsize=(12, 6))

# Plot Precision and Recall
plt.plot(epochs, train_precision, label='Train Precision', marker='o')
plt.plot(epochs, val_recall, label='Validation Recall', marker='o')

# Plot F1-Score
plt.plot(epochs, f1_score, label='F1-Score', marker='o', linestyle='--')

plt.title('Precision, Recall, and F1-Score over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()
