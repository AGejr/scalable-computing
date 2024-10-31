import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# Parameters
image_size = (128, 128)  # Target image size
data_dir = '/data/ir'  # Replace with your dataset path
output_dir = '/data/done'  # Base directory to save processed images
validation_split = 0.2    # Percentage of the data used for validation
random_seed = 42          # For reproducibility

# Step 1: Define Transformations
train_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])

# Step 2: Load Dataset
dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

# Step 3: Create Training and Validation Split
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# Create samplers for training and validation splits
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Step 4: Data Loaders
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

# Initialize lists to store the data
train_images, train_labels = [], []
val_images, val_labels = [], []

# Step 5: Process and Save Training Data in Memory
for images, labels in tqdm(train_loader, total=len(train_loader), desc="Processing Training Data"):
    train_images.append(images)
    train_labels.append(labels)

# Concatenate all batches into a single tensor
train_images = torch.cat(train_images)
train_labels = torch.cat(train_labels)

# Save training data to a .pth file
train_data_path = os.path.join(output_dir, 'train.pth')
torch.save({'images': train_images, 'labels': train_labels}, train_data_path)

# Step 6: Process and Save Validation Data in Memory
for images, labels in tqdm(val_loader, total=len(val_loader), desc="Processing Validation Data"):
    val_images.append(images)
    val_labels.append(labels)

# Concatenate all batches into a single tensor
val_images = torch.cat(val_images)
val_labels = torch.cat(val_labels)

# Save validation data to a .pth file
val_data_path = os.path.join(output_dir, 'val.pth')
torch.save({'images': val_images, 'labels': val_labels}, val_data_path)

os.system(f'gcloud storage cp {os.path.join(output_dir, "train.pth")} gs://ml-model-bucket-123456/train.pth')
os.system(f'gcloud storage cp {os.path.join(output_dir, "val.pth")} gs://ml-model-bucket-123456/val.pth')
