import torch
from torchvision import datasets, transforms
import os

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(45),transforms.ColorJitter(brightness=[0,0.1],contrast=[0,0.1],saturation=[0,0.1],hue=[0,0.1])])

# Download and load the training data
trainset = datasets.FashionMNIST('./data/F_MNIST_data/', download=True, train=True, transform=transform)

# Download and load the test data
testset = datasets.FashionMNIST('./data/F_MNIST_data/', download=True, train=False, transform=transform)

print("FashionMNIST dataset downloaded and saved successfully.")
# Define the directory to save the dataset
save_dir = './data'
os.makedirs(save_dir, exist_ok=True)

# Save the training data
torch.save(trainset, os.path.join(save_dir, 'trainset.pth'))

# Save the test data
torch.save(testset, os.path.join(save_dir, 'testset.pth'))

print(f"Training and test datasets saved to {save_dir}")

os.system('gcloud storage cp ./data/trainset.pth gs://ml-model-bucket-123456/trainset.pth')
os.system('gcloud storage cp ./data/testset.pth gs://ml-model-bucket-123456/testset.pth')

os.system('rm -rf ./data')
