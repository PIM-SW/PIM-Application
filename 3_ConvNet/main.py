import torch
from torch import nn
from torchsummary import summary

from torch.utils.data import DataLoader # wraps an iterable around the dataset
from torchvision import datasets # this module contains Dataset objects
from torchvision import transforms
from torchvision.transforms import ToTensor

import numpy as np
import argparse
# from resnet import ResNet18_baseline
from resnet import ResNet18

# Select Compute Device
# [SSH] Temporarily commented out for CPU execution
# train_on_gpu = torch.cuda.is_available()
train_on_gpu = False
device = "cuda" if train_on_gpu else "cpu"
if train_on_gpu:
    print("CUDA is available. Running on GPU...")
else:
    print("CUDA is NOT available. Running on CPU...")

# Parse Argument
parser = argparse.ArgumentParser(description="Run PyTorch implementation of ResNet")

parser.add_argument('--batch_size', type=int, required=True) # batch size for training and evaluation
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--eval_only', action='store_true')
args = parser.parse_args()

# Transformation (Pre-processing)
train_transformation = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomHorizontalFlip(),
])

test_transformation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CIFAR10 Dataset
## Train and Validation (80:20 split)
training_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=train_transformation)
train_data_size = (np.floor(len(training_dataset)*0.8)).astype(int)
val_data_size = len(training_dataset) - train_data_size
train_data, val_data = torch.utils.data.random_split(training_dataset, [train_data_size, val_data_size], generator=torch.Generator().manual_seed(123))
## Test
# test_data = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=test_transformation)

## [SSH] Used for inferencing only a certain portion of the test set
test_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=test_transformation)
sample_pct = 0.005
sample_data_size = (np.floor(len(test_dataset)*sample_pct)).astype(int)
unused_data_size = len(test_dataset) - sample_data_size
test_data, unused_data = torch.utils.data.random_split(test_dataset, [sample_data_size, unused_data_size], generator=torch.Generator().manual_seed(123))

# DataLoader
train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Train loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) # X: [B, C, W, H] Y: [B]
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"  Test Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Load ResNet model
model = ResNet18(num_classes=10).to(device)

# Loss Function and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-3)

# Train
if not args.eval_only:
    epochs = args.epochs
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        if t % 5 == 0:
            test(val_dataloader, model, loss_fn)

    print("Training Done! Saving trained model...")
    torch.save(model.state_dict(), "ResNet18_CIFAR10.pt")

# Evaluation
print("Running ResNet18...")
model.load_state_dict(torch.load('./ResNet18_CIFAR10.pt'))
model.eval()
test(test_dataloader, model, loss_fn)
