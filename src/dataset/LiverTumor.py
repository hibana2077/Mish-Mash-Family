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
from io import BytesIO
from sklearn.model_selection import train_test_split


class LiverTumorDataset(Dataset):
    """
    Liver Tumor Classification Dataset (PyTorch style)
    
    Similar to MNIST dataset structure, this dataset provides liver images
    for binary classification:
    - 0: Normal
    - 1: Tumor
    
    Args:
        root (str): Root directory containing the dataset
        train (bool): If True, creates dataset from training set, otherwise from testing set
        transform (callable, optional): A function/transform to apply to images
        target_transform (callable, optional): A function/transform to apply to targets
        download (bool): If True, downloads the dataset from HuggingFace
        train_split (float): Ratio for train/test split (default: 0.8)
        random_state (int): Random state for reproducible splits
    """
    
    # Dataset metadata
    classes = ['Normal', 'Tumor']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # HuggingFace dataset URL
    dataset_url = "https://huggingface.co/datasets/SofiaVouzika/Liver_Tumor/resolve/main/data/train-00000-of-00001.parquet?download=true"
    
    def __init__(self, root, train=True, transform=None, target_transform=None, 
                 download=False, train_split=0.8, random_state=42):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.train_split = train_split
        self.random_state = random_state
        
        if download:
            self.download()
            
        # Load and split data
        self.data, self.targets = self._load_data()
        
    def _load_data(self):
        """Load image data and labels from parquet file"""
        parquet_path = os.path.join(self.root, "train.parquet")
        
        if not os.path.exists(parquet_path):
            raise RuntimeError(
                f'Dataset not found at {parquet_path}. '
                f'You can use download=True to download it from HuggingFace'
            )
        
        # Load parquet file
        df = pd.read_parquet(parquet_path)
        
        # Extract images and labels
        images = df['image'].tolist()
        labels = df['label'].tolist()
        
        # Split into train/test
        train_images, test_images, train_labels, test_labels = train_test_split(
            images, labels, 
            train_size=self.train_split,
            random_state=self.random_state,
            stratify=labels  # Ensure balanced split
        )
        
        if self.train:
            return train_images, train_labels
        else:
            return test_images, test_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        image_data = self.data[idx]
        target = self.targets[idx]
        
        # Convert bytes to PIL Image
        image_bytes = image_data['bytes']
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return image, target
    
    def download(self):
        """Download the dataset from HuggingFace"""
        parquet_path = os.path.join(self.root, "train.parquet")
        
        if os.path.exists(parquet_path):
            print(f"Dataset already exists at {parquet_path}")
            return
            
        print(f"Downloading dataset from HuggingFace...")
        
        try:
            # Create root directory if it doesn't exist
            os.makedirs(self.root, exist_ok=True)
            
            # Download using curl
            subprocess.run([
                "curl", "-L", "-o", parquet_path, self.dataset_url
            ], check=True)
            
            print(f"Dataset downloaded successfully to {parquet_path}")
            
            # Verify download
            if os.path.exists(parquet_path):
                df = pd.read_parquet(parquet_path)
                print(f"Dataset loaded successfully with shape: {df.shape}")
            else:
                raise RuntimeError("Download completed but file not found")
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
        except Exception as e:
            raise RuntimeError(f"Error processing dataset: {e}")
    
    def get_class_distribution(self):
        """Get class distribution for current split"""
        class_counts = {}
        for target in self.targets:
            class_name = self.classes[target]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts
    
    def extra_repr(self):
        return f"Split: {'Train' if self.train else 'Test'}"


# Convenience functions (MNIST-like interface)
def make_liver_tumor_dataset(root, train=True, transform=None, target_transform=None, 
                           download=False, train_split=0.8, random_state=42):
    """Create LiverTumorDataset with MNIST-like interface"""
    return LiverTumorDataset(
        root=root,
        train=train, 
        transform=transform,
        target_transform=target_transform,
        download=download,
        train_split=train_split,
        random_state=random_state
    )


# Example usage and default transforms
class LiverTumorTransforms:
    """Common transforms for liver tumor dataset"""
    
    @staticmethod
    def get_train_transform():
        """Default training transforms with data augmentation"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),  # Medical images can benefit from vertical flip
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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
    train_dataset = LiverTumorDataset(
        root='./data',
        train=True,
        transform=LiverTumorTransforms.get_train_transform(),
        download=True,  # This will download the parquet file
        train_split=0.8,
        random_state=42
    )
    
    test_dataset = LiverTumorDataset(
        root='./data', 
        train=False,
        transform=LiverTumorTransforms.get_test_transform(),
        train_split=0.8,
        random_state=42
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
        class_counts = dataset.get_class_distribution()
        print(f"{split}:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
    
    # Example: Get a sample
    image, label = train_dataset[0]
    print(f"\nSample info:")
    print(f"Image shape: {image.shape}")
    print(f"Label: {label} ({train_dataset.classes[label]})")
    
    # Print train/test split info
    total_samples = len(train_dataset) + len(test_dataset)
    print(f"\nSplit info:")
    print(f"Total samples: {total_samples}")
    print(f"Train ratio: {len(train_dataset)/total_samples:.2f}")
    print(f"Test ratio: {len(test_dataset)/total_samples:.2f}")