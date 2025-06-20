import os
import tarfile
import requests
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json


class StanfordDogsDataset(Dataset):
    """
    Stanford Dogs Dataset - PyTorch Dataset class similar to MNIST
    
    Args:
        root (str): Root directory where dataset will be stored
        train (bool): If True, load training set, else test set
        transform (callable, optional): Transform to be applied on images
        target_transform (callable, optional): Transform to be applied on labels
        download (bool): If True, download dataset if not found
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        # Create directories
        os.makedirs(root, exist_ok=True)
        
        self.data_dir = os.path.join(root, 'Images')
        self.images_dir = self.data_dir
        
        if download:
            self._download()
            
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
            
        # Load class names and create label mapping
        self.classes = self._get_classes()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = self._make_dataset()
        
        # Split into train/test (80/20 split)
        split_idx = int(0.8 * len(self.samples))
        if self.train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
    
    def _download(self):
        """Download and extract the dataset"""
        if self._check_exists():
            print('Dataset already exists, skipping download')
            return
            
        print('Downloading Stanford Dogs Dataset...')
        
        url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
        tar_path = os.path.join(self.root, 'images.tar')
        
        try:
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(tar_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print('Download completed! Extracting...')
            
            # Extract tar file
            import tarfile
            with tarfile.open(tar_path, 'r') as tar:
                # Extract with progress bar
                members = tar.getmembers()
                with tqdm(desc="Extracting", total=len(members)) as pbar:
                    for member in members:
                        tar.extract(member, self.root)
                        pbar.update(1)
            
            # Clean up tar file
            os.remove(tar_path)
            print('Extraction completed successfully!')
            
        except Exception as e:
            print(f"Download failed: {e}")
            if os.path.exists(tar_path):
                os.remove(tar_path)
            raise
    
    def _check_exists(self):
        """Check if dataset exists"""
        return os.path.exists(self.images_dir)
    
    def _get_classes(self):
        """Get all dog breed classes from directory names"""
        if not os.path.exists(self.images_dir):
            return []
            
        classes = []
        for class_dir in sorted(os.listdir(self.images_dir)):
            if os.path.isdir(os.path.join(self.images_dir, class_dir)):
                # Extract breed name from directory name (remove ID prefix)
                breed_name = class_dir.split('-', 1)[1] if '-' in class_dir else class_dir
                classes.append(breed_name)
        return classes
    
    def _make_dataset(self):
        """Create list of (image_path, class_index) tuples"""
        samples = []
        
        if not os.path.exists(self.images_dir):
            return samples
            
        print("Loading dataset...")
        class_dirs = sorted(os.listdir(self.images_dir))
        
        for class_dir in tqdm(class_dirs, desc="Processing classes"):
            class_path = os.path.join(self.images_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            # Extract breed name and get class index
            breed_name = class_dir.split('-', 1)[1] if '-' in class_dir else class_dir
            class_idx = self.class_to_idx[breed_name]
            
            # Get all image files in this class directory
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        img_path, target = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return image, target
    
    def get_class_name(self, idx):
        """Get class name from class index"""
        return self.classes[idx]


# Example usage and transforms
def get_transforms(train=True):
    """Get default transforms for the dataset"""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


# Demo usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Create dataset
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)
    
    # Load training set
    train_dataset = StanfordDogsDataset(
        root='./data',
        train=True,
        transform=train_transform,
        download=True  # Set to True to download
    )
    
    # Load test set
    test_dataset = StanfordDogsDataset(
        root='./data',
        train=False,
        transform=test_transform,
        download=False  # Already downloaded
    )
    
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"First few classes: {train_dataset.classes[:5]}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:5]}")
    print(f"Sample class names: {[train_dataset.get_class_name(label.item()) for label in labels[:5]]}")
    
    # Show dataset statistics
    print("\nDataset Statistics:")
    print(f"Total classes: {len(train_dataset.classes)}")
    print(f"Sample class distribution (first 10):")
    class_counts = {}
    for _, label in train_dataset.samples:
        class_name = train_dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    for i, (cls_name, count) in enumerate(sorted(class_counts.items())[:10]):
        print(f"  {cls_name}: {count} images")