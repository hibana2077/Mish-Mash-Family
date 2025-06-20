import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np

# Import utilities
from utils.data_loader import get_dataloaders
from utils.model_utils import create_model_with_activation
from utils.training import train_one_epoch, evaluate
from utils.logging import (
    save_metrics, save_classification_report, 
    save_confusion_matrix, create_output_filename
)


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Load configuration
    cfg_path = os.path.join(os.path.dirname(__file__), 'main.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set random seed for reproducibility
    seed = cfg.get('seed', 42)
    set_random_seed(seed)
    print(f"Random seed set to: {seed}")

    # Extract configuration values
    model_name = cfg['model']['name']
    dataset_name = cfg['dataset']['name']
    activation_type = cfg['activation']['type']
    
    # Prepare logging with new naming convention
    save_dir = cfg['logging']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Device
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model (create first to get transforms)
    pretrained = cfg['model'].get('pretrained', False)
    # First create with default num_classes to get pretrained_cfg
    temp_model = timm.create_model(model_name, pretrained=pretrained)
    
    # Data (now with model-specific transforms)
    train_loader, test_loader = get_dataloaders(cfg, temp_model)
    num_classes = len(train_loader.dataset.classes) if hasattr(train_loader.dataset, 'classes') else len(test_loader.dataset.classes)
    
    # Get class names for better reporting
    class_names = None
    if hasattr(train_loader.dataset, 'classes'):
        class_names = train_loader.dataset.classes
    elif hasattr(test_loader.dataset, 'classes'):
        class_names = test_loader.dataset.classes
    
    # Create final model with correct num_classes and activation
    model = create_model_with_activation(cfg, num_classes)
    model.to(device)
    
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name} ({num_classes} classes)")
    print(f"Activation: {activation_type}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=cfg['training']['learning_rate'], 
                          weight_decay=cfg['training']['weight_decay'])

    # Training loop
    epochs = cfg['training']['epochs']
    history = []
    best_val_acc = 0.0
    best_y_true = None
    best_y_pred = None
    best_model = None
    
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
        
        history.append({
            'epoch': epoch, 
            'train_loss': train_loss, 
            'train_acc': train_acc, 
            'val_loss': val_loss, 
            'val_acc': val_acc
        })
        
        # Update best validation accuracy and save best predictions
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_y_true = y_true
            best_y_pred = y_pred
            best_model = model

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")

    # Save results with new naming convention
    base_filename = create_output_filename(model_name, dataset_name, activation_type, "")
    
    # Save metrics
    metrics_path = os.path.join(save_dir, create_output_filename(model_name, dataset_name, activation_type, "csv"))
    save_metrics(history, metrics_path)
    print(f"Metrics saved to: {metrics_path}")
    
    # Save classification report (replacing confusion matrix as primary evaluation)
    report_path = os.path.join(save_dir, f"{base_filename}classification_report.csv")
    save_classification_report(best_y_true, best_y_pred, class_names, report_path)
    print(f"Classification report saved to: {report_path}")
    
    # Still save confusion matrix for reference
    cm_path = os.path.join(save_dir, f"{base_filename}confusion_matrix.png")
    save_confusion_matrix(best_y_true, best_y_pred, class_names, cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    print(f"\nAll results saved with prefix: {base_filename}")
    
    if cfg['activation']['type'] == 'moment':
        print(best_model)


if __name__ == '__main__':
    main()