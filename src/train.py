import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ConvLSTM
from torch.utils.tensorboard import SummaryWriter
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

BATCH_SIZE = 8
SEQUENCE_LENGTH = 5


def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    