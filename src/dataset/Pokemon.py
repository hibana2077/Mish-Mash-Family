import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from typing import Optional, Callable, Tuple, List
import requests
import zipfile
import shutil
from pathlib import Path

class PokemonDataset(Dataset):
    """
    Pokemon Classification Dataset - similar to MNIST structure
    
    Args:
        root (str): Root directory path where data will be stored
        split (str): 'train', 'test', or 'valid'
        transform (callable, optional): Optional transform to be applied on images
        target_transform (callable, optional): Optional transform to be applied on labels
        download (bool): If True, downloads the dataset from the internet
    """
    
    urls = {
        'train': 'https://huggingface.co/datasets/fcakyon/pokemon-classification/resolve/main/data/train.zip?download=true',
        'test': 'https://huggingface.co/datasets/fcakyon/pokemon-classification/resolve/main/data/test.zip?download=true',
        'valid': 'https://huggingface.co/datasets/fcakyon/pokemon-classification/resolve/main/data/valid.zip?download=true'
    }
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        if download:
            self.download()
        
        if not self._check_exists():
            raise RuntimeError(f'Dataset not found for split {split}. '
                             'You can use download=True to download it')
        
        # 載入圖片路徑和標籤
        self.images, self.labels = self._load_data()
        
        # 創建類別名稱到索引的映射
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(set(self.labels)))}
        self.classes = list(self.class_to_idx.keys())
        
        # 將標籤轉換為數字索引
        self.label_indices = [self.class_to_idx[label] for label in self.labels]
    
    def _check_exists(self) -> bool:
        """檢查數據集是否存在"""
        split_dir = self.root / self.split
        return split_dir.exists() and len(list(split_dir.glob('*/*.jpg'))) > 0
    
    def download(self):
        """下載並解壓數據集"""
        if self._check_exists():
            print(f'{self.split} split already exists, skipping download.')
            return
        
        print(f'Downloading {self.split} split...')
        
        # 創建根目錄
        self.root.mkdir(parents=True, exist_ok=True)
        
        # 下載zip文件
        zip_path = self.root / f'{self.split}.zip'
        url = self.urls[self.split]
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # 解壓文件到臨時目錄
        print(f'Extracting {self.split} split...')
        temp_extract_path = self.root / f'temp_{self.split}'
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)
        
        # 創建split目錄並移動文件
        split_dir = self.root / self.split
        split_dir.mkdir(exist_ok=True)
        
        # 移動所有pokemon資料夾到split目錄下
        for pokemon_dir in temp_extract_path.iterdir():
            if pokemon_dir.is_dir():
                target_dir = split_dir / pokemon_dir.name
                if target_dir.exists():
                    # 如果目標目錄已存在，移動所有文件
                    import shutil
                    for file in pokemon_dir.glob('*.jpg'):
                        shutil.move(str(file), str(target_dir))
                    pokemon_dir.rmdir()
                else:
                    pokemon_dir.rename(target_dir)
        
        # 清理臨時目錄和zip文件
        if temp_extract_path.exists():
            import shutil
            shutil.rmtree(temp_extract_path)
        zip_path.unlink()
        
        print(f'{self.split} split downloaded and extracted successfully!')
    
    def _load_data(self) -> Tuple[List[str], List[str]]:
        """載入圖片路徑和對應的標籤"""
        split_dir = self.root / self.split
        
        images = []
        labels = []
        
        # 只處理.jpg文件
        for img_path in split_dir.glob('*/*.jpg'):
            pokemon_name = img_path.parent.name
            images.append(str(img_path))
            labels.append(pokemon_name)
        
        return images, labels
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, target) where target is the class index
        """
        img_path = self.images[idx]
        target = self.label_indices[idx]
        
        # 載入圖片
        image = Image.open(img_path).convert('RGB')
        
        # 應用transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    def get_class_name(self, idx: int) -> str:
        """根據類別索引獲取類別名稱"""
        return self.classes[idx]
    
    def get_class_distribution(self) -> dict:
        """獲取每個類別的樣本數量"""
        from collections import Counter
        return dict(Counter(self.labels))


# 使用範例和輔助函數
def get_default_transforms(train: bool = True, image_size: int = 224):
    """獲取預設的數據增強transforms"""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(root: str, batch_size: int = 32, num_workers: int = 4, download: bool = False):
    """創建train, validation, test的DataLoader"""
    
    # 創建datasets
    train_dataset = PokemonDataset(
        root=root,
        split='train',
        transform=get_default_transforms(train=True),
        download=download
    )
    
    val_dataset = PokemonDataset(
        root=root,
        split='valid',
        transform=get_default_transforms(train=False),
        download=download
    )
    
    test_dataset = PokemonDataset(
        root=root,
        split='test',
        transform=get_default_transforms(train=False),
        download=download
    )
    
    # 創建dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes


# 使用範例
if __name__ == "__main__":
    # 基本使用
    root_dir = "./pokemon_data"
    
    # 創建dataset（自動下載）
    train_dataset = PokemonDataset(
        root=root_dir,
        split='train',
        transform=get_default_transforms(train=True),
        download=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes[:10]}...")  # 顯示前10個類別
    
    # 查看數據分布
    distribution = train_dataset.get_class_distribution()
    print(f"Class distribution: {list(distribution.items())[:5]}...")  # 顯示前5個
    
    # 獲取一個樣本
    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label} ({train_dataset.get_class_name(label)})")
    
    # 創建所有dataloaders
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        root=root_dir,
        batch_size=32,
        download=True
    )
    
    print(f"\nDataLoader info:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 測試一個batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break