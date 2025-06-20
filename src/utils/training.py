"""
Training utilities for model training and evaluation.
"""
import torch
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='Training', leave=False)
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)
        
        pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
    
    return running_loss/total, correct/total


def evaluate(model, loader, criterion, device):
    """
    Evaluate the model.
    
    Args:
        model: The model to evaluate
        loader: Test data loader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        tuple: (average_loss, accuracy, true_labels, predictions)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    pbar = tqdm(loader, desc='Evaluating', leave=False)
    
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += inputs.size(0)
            
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
    
    return running_loss/total, correct/total, y_true, y_pred