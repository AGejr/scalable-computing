import torch
from torchvision import datasets, transforms
import os

# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.RandomRotation(10),       
    transforms.RandomCrop(28, padding=4), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Download and load the training data
train_valid_dataset = datasets.FashionMNIST('./data/F_MNIST_data/', download=True, train=True, transform=transform)

valid_ratio = 0.2
nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
nb_valid =  int(valid_ratio * len(train_valid_dataset))
train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])

print("FashionMNIST dataset downloaded and saved successfully.")
# Define the directory to save the dataset
save_dir = './data'
os.makedirs(save_dir, exist_ok=True)

# Save the training data
torch.save(train_dataset, os.path.join(save_dir, 'trainset.pth'))

# Save the test data
torch.save(valid_dataset, os.path.join(save_dir, 'valset.pth'))

print(f"Training and test datasets saved to {save_dir}")

os.system('gcloud storage cp ./data/trainset.pth gs://ml-model-bucket-123456/trainset.pth')
os.system('gcloud storage cp ./data/testset.pth gs://ml-model-bucket-123456/testset.pth')

os.system('rm -rf ./data')
