# training/federated.py
import copy
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def split_dataset(dataset, num_clients: int) -> List[Subset]:
    n = len(dataset)
    lengths = [n // num_clients] * num_clients
    for i in range(n % num_clients):
        lengths[i] += 1
    return torch.utils.data.random_split(dataset, lengths)


def train_one_client(model: nn.Module,
                     dataloader: DataLoader,
                     device,
                     epochs: int = 1,
                     lr: float = 1e-4) -> Dict:
    model = copy.deepcopy(model)
    model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def fed_avg(state_dicts: List[Dict], weights: List[float]) -> Dict:
    global_state = copy.deepcopy(state_dicts[0])
    for key in global_state.keys():
        global_state[key] = state_dicts[0][key] * weights[0]
        for i in range(1, len(state_dicts)):
            global_state[key] += state_dicts[i][key] * weights[i]
    return global_state


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total if total > 0 else 0
    return acc
