import os
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class BrainTumorDataset(Dataset):
    """
    Brain Tumor Classification Dataset (PyTorch style)
    
    Similar to MNIST dataset structure, this dataset provides brain X-ray images
    for fine-grain classification into 4 categories:
    - Glioma Tumor (n0)
    - Meningioma Tumor (n1) 
    - Pituitary Tumor (n2)
    - No Tumor (n3)
    
    Args:
        root (str): Root directory containing the dataset
        train (bool): If True, creates dataset from training set, otherwise from validation set
        transform (callable, optional): A function/transform to apply to images
        target_transform (callable, optional): A function/transform to apply to targets
        download (bool): If True, downloads the dataset (not implemented in this version)
    """
    
    # Dataset metadata
    classes = [
        'Glioma Tumor',
        'Meningioma Tumor', 
        'Pituitary Tumor',
        'No Tumor'
    ]
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    url = "https://ibm.ent.box.com/index.php?rm=box_download_shared_file&shared_name=5ich3fqgpnbmkdho2eoe7fe4uwrplcfi&file_id=f_978363130854"
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        if download:
            self.download()
            
        # Determine data path based on train/test split
        if self.train:
            data_path = os.path.join(self.root, "xrays", "training", "training")
        else:
            data_path = os.path.join(self.root, "xrays", "validation", "validation") 
            
        if not os.path.exists(data_path):
            raise RuntimeError(
                f'Dataset not found at {data_path}. '
                f'You can use download=True to download it (manual download required)'
            )
            
        # Load image paths and labels
        self.data, self.targets = self._load_data(data_path)
        
    def _load_data(self, filepath):
        """Load image paths and corresponding labels"""
        paths = list(Path(filepath).glob("**/*.jpg"))
        data = []
        
        for path in paths:
            # Extract tumor type from folder name (n0, n1, n2, n3)
            tumor_folder = str(path).split(os.sep)[-2]
            label_idx = int(tumor_folder[1])  # Extract number from 'n0', 'n1', etc.
            label_name = self.classes[label_idx]
            
            data.append({
                "file": str(path), 
                "label": label_name,
                "label_idx": label_idx
            })
        
        # Convert to DataFrame and sort for consistent ordering
        df = pd.DataFrame(data)
        df = df.sort_values("file").reset_index(drop=True)
        
        image_paths = df["file"].tolist()
        targets = df["label_idx"].tolist()
        
        return image_paths, targets
    
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
        """Download the dataset (placeholder - requires manual download)"""
        print(f"Please manually download the dataset from: {self.url}")
        print(f"Extract it to: {self.root}")
        raise NotImplementedError(
            "Automatic download not implemented. "
            "Please download manually from Kaggle and extract to the root directory."
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
    # Create datasets
    train_dataset = BrainTumorDataset(
        root='./data',
        train=True,
        transform=BrainTumorTransforms.get_train_transform()
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
    
    # Example: Get a sample
    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label} ({train_dataset.classes[label]})")