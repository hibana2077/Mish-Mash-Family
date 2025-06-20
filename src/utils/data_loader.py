"""
Data loading utilities for different datasets.
"""
import timm
from torch.utils.data import DataLoader

# Import dataset utilities
from dataset.CUB200 import create_cub_dataloaders, CUB200Dataset
from dataset.IP102 import create_ip102_dataloaders, IP102Dataset
from dataset.StandfordDogs import StanfordDogsDataset, get_transforms
from dataset.Pokemon import PokemonDataset, create_dataloaders
from dataset.BrainTumor import BrainTumorDataset, BrainTumorTransforms


def get_dataloaders(cfg, model=None):
    """
    Create data loaders for the specified dataset.
    
    Args:
        cfg: Configuration dictionary
        model: Model to get transforms from (optional)
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    name = cfg['dataset']['name']
    root = cfg['dataset']['root']
    batch_size = cfg['dataset']['batch_size']
    num_workers = cfg['dataset']['num_workers']
    
    # Get timm transforms if model is provided
    if model is not None:
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        test_transform = timm.data.create_transform(**data_cfg)
        # For training, add data augmentation
        train_data_cfg = data_cfg.copy()
        train_data_cfg.update({'is_training': True})
        train_transform = timm.data.create_transform(**train_data_cfg)
    else:
        # Fallback to default transforms
        train_transform = test_transform = None
    
    if name == 'CUB200':
        if train_transform is not None:
            # Create custom dataloaders with timm transforms
            train_ds = CUB200Dataset(root, train=True, transform=train_transform, download=True)
            test_ds = CUB200Dataset(root, train=False, transform=test_transform, download=False)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                    num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                                   num_workers=num_workers, pin_memory=True)
        else:
            train_loader, test_loader = create_cub_dataloaders(root, batch_size, num_workers)
            
    elif name == 'IP102':
        if train_transform is not None:
            # Create custom dataloaders with timm transforms
            train_ds = IP102Dataset(root, split='train', transform=train_transform, download=True)
            test_ds = IP102Dataset(root, split='test', transform=test_transform, download=False)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                    num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                                   num_workers=num_workers, pin_memory=True)
        else:
            dl = create_ip102_dataloaders(root, batch_size, num_workers, download=True)
            train_loader, test_loader = dl['train'], dl['test']
            
    elif name == 'StanfordDogs':
        if train_transform is not None:
            train_ds = StanfordDogsDataset(root, train=True, transform=train_transform, download=True)
            test_ds = StanfordDogsDataset(root, train=False, transform=test_transform, download=False)
        else:
            train_ds = StanfordDogsDataset(root, train=True, transform=get_transforms(True), download=True)
            test_ds = StanfordDogsDataset(root, train=False, transform=get_transforms(False), download=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=True)
    elif name == 'Pokemon':
        if train_transform is not None:
            train_ds = PokemonDataset(root, split='train', transform=train_transform, download=True)
            test_ds = PokemonDataset(root, split='test', transform=test_transform, download=False)
        else:
            train_ds = PokemonDataset(root, split='train', transform=get_transforms(True), download=True)
            test_ds = PokemonDataset(root, split='test', transform=get_transforms(False), download=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=True)
    elif name == 'BrainTumor':
        if train_transform is not None:
            train_ds = BrainTumorDataset(root, train=True, transform=train_transform, download=True)
            test_ds = BrainTumorDataset(root, train=False, transform=test_transform, download=False)
        else:
            train_ds = BrainTumorDataset(root, train=True, transform=BrainTumorTransforms.get_train_transform(), download=True)
            test_ds = BrainTumorDataset(root, train=False, transform=BrainTumorTransforms.get_test_transform(), download=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=True)
    else:
        raise ValueError(f"Unsupported dataset {name}")
        
    return train_loader, test_loader