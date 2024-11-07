import os
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
import numpy as np

# Parameters
image_size = (128, 128)  # Target image size
data_dir = '/data/ir'  # Replace with your dataset path
output_dir = '/data/done'  # Base directory to save processed images
validation_split = 0.2    # Percentage of the data used for validation
random_seed = 42          # For reproducibility
cloud_bucket_path = 'gs://ml-model-bucket-123456'  # Cloud bucket path

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Step 1: Define Transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# Step 2: Load Dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Step 3: Create Training and Validation Split
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# Use Subset to split the dataset
train_set = Subset(dataset, train_indices)
val_set = Subset(dataset, val_indices)

# Step 4: Save Training and Validation Data to .pth Files
train_data_path = os.path.join(output_dir, 'trainset.pth')
val_data_path = os.path.join(output_dir, 'valset.pth')

# Save the datasets directly
torch.save(train_set, train_data_path)
torch.save(val_set, val_data_path)

print(f"Training and validation datasets saved to {output_dir}")
