from __future__ import print_function

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Conv layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 32 filters, small kernel
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                 # Pooling to reduce spatial size
        self.dropout1 = nn.Dropout(0.3)

        # Conv layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 64 filters
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)

        # Conv layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 128 filters
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.3)

        # Flatten and Fully Connected Layer
        # After pooling, spatial size ~ (128, 3, 3) for input of (1, 30, 30)
        self.flatten_size = 128 * 3 * 3
        self.fc = nn.Linear(self.flatten_size, 5)

    def forward(self, x):
        x = x / 255.0  # Rescale input
        
        # Apply layers with ReLU activation, pooling, and dropout
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten for the fully connected layer
        x = x.view(-1, self.flatten_size)
        x = self.fc(x)
        return torch.sigmoid(x)  # Sigmoid for binary classification

def train(args, model, device, train_loader, epoch, writer):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=1e-4)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        # Use binary cross-entropy for binary classification
        target = target.float()  # Ensure target is float for BCELoss
        loss = F.binary_cross_entropy(output, target.unsqueeze(1))  # Use BCELoss

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar("loss", loss.item(), niter)


def test(model, device, test_loader, writer, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = (output >= 0.5).float()  # Binary thresholding for accuracy
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = float(correct) / len(test_loader.dataset)
    print("\naccuracy={:.4f}\n".format(accuracy))
    writer.add_scalar("accuracy", accuracy, epoch)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch FashionMNIST Example")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=1, metavar="N", help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
    parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
    parser.add_argument("--dir", default="logs", metavar="L", help="directory where summary logs are stored")
    parser.add_argument("--backend", type=str, help="Distributed backend", choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI], default=dist.Backend.GLOO)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")
        if args.backend != dist.Backend.NCCL:
            print("Warning. Please use `nccl` distributed backend for the best performance using GPUs")

    writer = SummaryWriter(args.dir)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)

    print("Using distributed PyTorch with {} backend".format(args.backend))
    if "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "1234"

    print(f"World Size: {os.environ['WORLD_SIZE']}. Rank: {os.environ['RANK']}")

    dist.init_process_group(backend=args.backend)
    model = nn.parallel.DistributedDataParallel(model)

    gcs_mount_point = os.getenv('GCS_MOUNT_POINT', '/data')
    train_data = torch.load(os.path.join(gcs_mount_point, 'trainset.pth'))
    test_data = torch.load(os.path.join(gcs_mount_point, 'testset.pth'))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=DistributedSampler(train_data))
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, sampler=DistributedSampler(test_data))

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, epoch, writer)
        test(model, device, test_loader, writer, epoch)

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(gcs_mount_point, "mnist_cnn.pt"))

if __name__ == "__main__":
    main()
