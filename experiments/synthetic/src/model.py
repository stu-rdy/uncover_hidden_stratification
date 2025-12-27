import torch
import torch.nn as nn
import torchvision
import numpy as np
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score

def get_model(arch='resnet50', num_classes=10, pretrained=True):
    if arch == 'resnet50':
        model = torchvision.models.resnet50(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(2048, num_classes)
    elif arch == 'resnet18':
        model = torchvision.models.resnet18(weights='DEFAULT' if pretrained else None)
        model.fc = nn.Linear(512, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    return model

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for (x, y, g), _ in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_groups = []
    all_names = []
    
    with torch.no_grad():
        for ((x, y, g), name) in tqdm(dataloader, desc="Evaluating"):
            x = x.to(device)
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.numpy())
            all_groups.extend(g.numpy())
            all_names.extend(name)
            
    return {
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "groups": np.array(all_groups),
        "names": np.array(all_names),
        "acc": balanced_accuracy_score(all_labels, all_preds)
    }
