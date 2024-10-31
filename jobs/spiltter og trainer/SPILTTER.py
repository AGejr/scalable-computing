import os
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from PIL import Image  # For saving images
import numpy as np
from tqdm import tqdm  # For progress bar

# Parameters
image_size = (128, 128)  # Target image size
data_dir = 'C:\\Users\\mglad\\Documents\\lip_pose_pic\\ir'  # Replace with your dataset path
output_dir = 'c:\\Users\\mglad\\Documents\\lip_pose_pic\\done'  # Base directory to save processed images
validation_split = 0.2   # Percentage of the data used for validation
random_seed = 42         # For reproducibility

# Step 1: Define Transformations
train_transforms = transforms.Compose([
    transforms.Resize(image_size),             # Resize to image_size
    transforms.RandomRotation(20),             # Random rotation up to 20 degrees
    transforms.RandomHorizontalFlip(),         # Random horizontal flip
    transforms.ToTensor(),                     # Convert image to tensor
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize to ImageNet standards
])

# Validation/test transformation: Only resizing and normalization
test_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Step 2: Load Dataset
dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)

# Step 3: Create Output Directories
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Step 4: Create Training and Validation Split
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# Create samplers for training and validation splits
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

# Step 5: Data Loaders
train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

# Step 6: Save Processed Images for Train Set
for i, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
    for j, (img, label) in enumerate(zip(images, labels)):
        img = img.clone().cpu()  # Clone the tensor and move to CPU
        img = img.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
        img = (img * 255).numpy().astype(np.uint8)  # Denormalize and convert to uint8
        
        # Get class name from dataset
        class_name = dataset.classes[label.item()]
        class_dir = os.path.join(train_dir, class_name)
        
        # Create class directory if it doesn't exist
        os.makedirs(class_dir, exist_ok=True)

        # Create a filename based on the index and label
        save_path = os.path.join(class_dir, f"train_img_{i * 32 + j}.png")
        # Save image using PIL
        image_pil = Image.fromarray(img)
        image_pil.save(save_path)

# Step 7: Save Processed Images for Validation Set
for i, (images, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
    for j, (img, label) in enumerate(zip(images, labels)):
        img = img.clone().cpu()  # Clone the tensor and move to CPU
        img = img.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
        img = (img * 255).numpy().astype(np.uint8)  # Denormalize and convert to uint8
        
        # Get class name from dataset
        class_name = dataset.classes[label.item()]
        class_dir = os.path.join(val_dir, class_name)
        
        # Create class directory if it doesn't exist
        os.makedirs(class_dir, exist_ok=True)

        # Create a filename based on the index and label
        save_path = os.path.join(class_dir, f"val_img_{i * 32 + j}.png")
        
        # Save image using PIL
        image_pil = Image.fromarray(img)
        image_pil.save(save_path)

print("Processed images saved successfully!")
