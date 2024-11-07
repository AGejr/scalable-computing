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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, epoch, writer, train_losses, train_accuracies):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    correct = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate the number of correct predictions for the batch
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        

    average_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / len(train_loader.dataset)
    writer.add_scalar("train_loss", average_loss, epoch)
    writer.add_scalar("train_accuracy", accuracy, epoch)
    train_losses.append(average_loss)
    train_accuracies.append(accuracy)
    print(f"Train Epoch: {epoch} Loss: {loss.item():.4f} Accuracy: {accuracy:.2f}%")

def test(model, device, test_loader, writer, epoch, test_losses, test_accuracies):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    writer.add_scalar("test_loss", test_loss, epoch)
    writer.add_scalar("test_accuracy", accuracy, epoch)
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch FashionMNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--save-summaries",
        action="store_true",
        default=True,
        help="For Saving training summaries",
    )
    parser.add_argument(
        "--backend",
        type=str,
        help="Distributed backend",
        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
        default=dist.Backend.GLOO,
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")
        if args.backend != dist.Backend.NCCL:
            print(
                "Warning. Please use `nccl` distributed backend for the best performance using GPUs"
            )
    
    # Get GCS mount point from environment variable
    gcs_mount_point = os.getenv('GCS_MOUNT_POINT', '/data')

    # Create a directory with the current timestamp
    output_dir = f"{gcs_mount_point}/dist-mnist"
    os.makedirs(output_dir, exist_ok=True)

    if args.save_summaries:
        writer = SummaryWriter(output_dir)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Attach model to the device.
    model = Net().to(device)

    print("Using distributed PyTorch with {} backend".format(args.backend))
    # Set distributed training environment variables to run this training script locally.
    if "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "1234"

    print(f"World Size: {os.environ['WORLD_SIZE']}. Rank: {os.environ['RANK']}")

    dist.init_process_group(backend=args.backend)
    model = nn.parallel.DistributedDataParallel(model)

    # Load train and test datasets from .pth files
    train_data = torch.load(os.path.join(gcs_mount_point, 'trainset.pth'))
    test_data = torch.load(os.path.join(gcs_mount_point, 'testset.pth'))

    # Add train and test loaders.
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=DistributedSampler(train_data),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.test_batch_size,
        sampler=DistributedSampler(test_data),
    )

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, epoch, writer, train_losses, train_accuracies)
        test(model, device, test_loader, writer, epoch, test_losses, test_accuracies)

    global_rank = dist.get_global_rank(model.process_group, dist.get_rank(model.process_group))
    print(f"Global Rank: {global_rank}")
    
    if args.save_model and global_rank == 0:
        torch.save(model.state_dict(), os.path.join(output_dir, "mnist_cnn.pt"))

if __name__ == "__main__":
    main()
