import os
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import subprocess
import shutil


class BrainTumorDataset(Dataset):
    """
    Brain Tumor Classification Dataset (PyTorch style)
    
    Similar to MNIST dataset structure, this dataset provides brain X-ray images
    for fine-grain classification into 4 categories:
    - glioma_tumor
    - meningioma_tumor  
    - pituitary_tumor
    - no_tumor
    
    Args:
        root (str): Root directory containing the dataset
        train (bool): If True, creates dataset from training set, otherwise from testing set
        transform (callable, optional): A function/transform to apply to images
        target_transform (callable, optional): A function/transform to apply to targets
        download (bool): If True, downloads the dataset from GitHub
    """
    
    # Dataset metadata
    classes = [
        'glioma_tumor',
        'meningioma_tumor', 
        'pituitary_tumor',
        'no_tumor'
    ]
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # GitHub repository URL
    git_url = "https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet.git"
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        if download:
            self.download()
            
        # Determine data path based on train/test split
        split_name = "Training" if self.train else "Testing"
        data_path = os.path.join(self.root, "Brain-Tumor-Classification-DataSet", split_name)
            
        if not os.path.exists(data_path):
            raise RuntimeError(
                f'Dataset not found at {data_path}. '
                f'You can use download=True to download it from GitHub'
            )
            
        # Load image paths and labels
        self.data, self.targets = self._load_data(data_path)
        
    def _load_data(self, data_path):
        """Load image paths and corresponding labels from class folders"""
        image_paths = []
        targets = []
        
        # Iterate through each class folder
        for class_name in self.classes:
            class_folder = os.path.join(data_path, class_name)
            
            if not os.path.exists(class_folder):
                print(f"Warning: Class folder {class_folder} not found, skipping...")
                continue
                
            # Get all jpg files in the class folder
            class_images = list(Path(class_folder).glob("*.jpg"))
            
            # Add images and labels
            for img_path in class_images:
                image_paths.append(str(img_path))
                targets.append(self.class_to_idx[class_name])
        
        # Sort for consistent ordering
        combined = list(zip(image_paths, targets))
        combined.sort(key=lambda x: x[0])  # Sort by image path
        
        image_paths, targets = zip(*combined) if combined else ([], [])
        
        return list(image_paths), list(targets)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path = self.data[idx]
        target = self.targets[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return image, target
    
    def download(self):
        """Download the dataset from GitHub repository"""
        dataset_path = os.path.join(self.root, "Brain-Tumor-Classification-DataSet")
        
        if os.path.exists(dataset_path):
            print(f"Dataset already exists at {dataset_path}")
            return
            
        print(f"Downloading dataset from {self.git_url}")
        
        try:
            # Create root directory if it doesn't exist
            os.makedirs(self.root, exist_ok=True)
            
            # Clone the repository directly to the desired location
            subprocess.run([
                "git", "clone", self.git_url, "Brain-Tumor-Classification-DataSet"
            ], check=True, cwd=self.root)
            
            print(f"Dataset downloaded successfully to {dataset_path}")
            
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
        except FileNotFoundError:
            raise RuntimeError(
                "Git is not installed or not found in PATH. "
                "Please install Git or download the dataset manually from: "
                f"{self.git_url}"
            )
    
    def extra_repr(self):
        return f"Split: {'Train' if self.train else 'Test'}"


# Convenience functions (MNIST-like interface)
def make_brain_tumor_dataset(root, train=True, transform=None, target_transform=None, download=False):
    """Create BrainTumorDataset with MNIST-like interface"""
    return BrainTumorDataset(
        root=root,
        train=train, 
        transform=transform,
        target_transform=target_transform,
        download=download
    )


# Example usage and default transforms
class BrainTumorTransforms:
    """Common transforms for brain tumor dataset"""
    
    @staticmethod
    def get_train_transform():
        """Default training transforms with data augmentation"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def get_test_transform():
        """Default test transforms without augmentation"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


# Example usage
if __name__ == "__main__":
    # Create datasets (with automatic download)
    train_dataset = BrainTumorDataset(
        root='./data',
        train=True,
        transform=BrainTumorTransforms.get_train_transform(),
        download=True  # This will clone the GitHub repository
    )
    
    test_dataset = BrainTumorDataset(
        root='./data', 
        train=False,
        transform=BrainTumorTransforms.get_test_transform()
    )
    
    # Create data loaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32, 
        shuffle=False,
        num_workers=4
    )
    
    # Print dataset info
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    
    # Print class distribution
    print("\nClass distribution:")
    for split, dataset in [("Train", train_dataset), ("Test", test_dataset)]:
        class_counts = {}
        for target in dataset.targets:
            class_name = dataset.classes[target]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"{split}:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
    
    # Example: Get a sample
    image, label = train_dataset[0]
    print(f"\nSample info:")
    print(f"Image shape: {image.shape}")
    print(f"Label: {label} ({train_dataset.classes[label]})")