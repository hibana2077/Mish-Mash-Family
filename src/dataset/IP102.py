import os
import tarfile
import urllib.request
from typing import Callable, Optional, Tuple, Any
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class IP102Dataset(Dataset):
    """IP102 Dataset for insect pest recognition.
    
    Args:
        root (str): Root directory where the dataset will be stored.
        split (str): Dataset split, one of 'train', 'val', 'test'.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    
    url = "https://huggingface.co/datasets/hibana2077/IP102/resolve/main/ip102_v1.1.tar?download=true"
    filename = "ip102_v1.1.tar"
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be one of ['train', 'val', 'test'], got {split}")
        
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Create root directory if it doesn't exist
        os.makedirs(self.root, exist_ok=True)
        
        if download:
            self.download()
        
        if not self._check_exists():
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it'
            )
        
        # Load the data
        self.data, self.targets = self._load_data()
    
    def _check_exists(self) -> bool:
        """Check if the dataset files exist."""
        extracted_dir = os.path.join(self.root, "ip102_v1.1")
        images_dir = os.path.join(extracted_dir, "images")
        split_file = os.path.join(extracted_dir, f"{self.split}.txt")
        
        return (
            os.path.isdir(images_dir) and
            os.path.isfile(split_file)
        )
    
    def download(self) -> None:
        """Download and extract the IP102 dataset."""
        if self._check_exists():
            print("Dataset already exists, skipping download.")
            return
        
        print("Downloading IP102 dataset...")
        
        # Download the tar file
        tar_path = os.path.join(self.root, self.filename)
        
        try:
            urllib.request.urlretrieve(self.url, tar_path)
            print(f"Downloaded {self.filename}")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
        
        # Extract the tar file
        print("Extracting dataset...")
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(self.root)
            print("Extraction completed")
        except Exception as e:
            raise RuntimeError(f"Failed to extract dataset: {e}")
        finally:
            # Clean up the tar file
            if os.path.exists(tar_path):
                os.remove(tar_path)
                print(f"Removed {self.filename}")
    
    def _load_data(self) -> Tuple[list, list]:
        """Load image paths and labels from the split file."""
        extracted_dir = os.path.join(self.root, "ip102_v1.1")
        images_dir = os.path.join(extracted_dir, "images")
        split_file = os.path.join(extracted_dir, f"{self.split}.txt")
        
        data = []
        targets = []
        
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        img_name = parts[0]
                        label = int(parts[1])
                        img_path = os.path.join(images_dir, img_name)
                        
                        if os.path.exists(img_path):
                            data.append(img_path)
                            targets.append(label)
                        else:
                            print(f"Warning: Image {img_path} not found")
        
        print(f"Loaded {len(data)} images for {self.split} split")
        return data, targets
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
            
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path = self.data[index]
        target = self.targets[index]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    @property
    def classes(self) -> list:
        """Get the list of class names (labels)."""
        # IP102 has 102 classes (0-101)
        return list(range(102))
    
    @property
    def class_to_idx(self) -> dict:
        """Get the mapping from class names to indices."""
        return {str(i): i for i in range(102)}


# Example usage and utility functions
def get_ip102_transforms(split='train'):
    """Get recommended transforms for IP102 dataset."""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_ip102_dataloaders(root_dir, batch_size=32, num_workers=4, download=False):
    """Create DataLoaders for all splits of IP102 dataset."""
    from torch.utils.data import DataLoader
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        dataset = IP102Dataset(
            root=root_dir,
            split=split,
            transform=get_ip102_transforms(split),
            download=download and split == 'train'  # Only download once
        )
        
        shuffle = split == 'train'
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders


if __name__ == "__main__":
    # Example usage
    
    # Create dataset with automatic download
    dataset = IP102Dataset(
        root="./data",
        split="train",
        transform=get_ip102_transforms('train'),
        download=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    
    # Get a sample
    image, label = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    
    # Create DataLoaders for all splits
    dataloaders = create_ip102_dataloaders(
        root_dir="./data",
        batch_size=32,
        num_workers=4,
        download=False  # Already downloaded above
    )
    
    # Test the DataLoaders
    for split, dataloader in dataloaders.items():
        batch = next(iter(dataloader))
        images, labels = batch
        print(f"{split} - Batch shape: {images.shape}, Labels shape: {labels.shape}")