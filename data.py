import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import preprocess_train, preprocess_test

class CustomDataset(Dataset):
    """Custom Dataset for image loading and processing using PyTorch operations."""

    def __init__(self, img_path,  transform=None):
        self.img_path = img_path
        self.image_size = 176
        # Read and store image paths from the dataset file
        with open(self.img_path, 'r') as f:
            self.record_list = [line.strip() for line in f]

        self.transform = transform if transform else self.default_transforms()

    def __len__(self):
        """Return the total number of samples."""
        return len(self.record_list)

    def default_transforms(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # Resize to image_size
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.ToTensor(),  # Convert image to tensor
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (ImageNet stats)
        ])

    def __getitem__(self, idx):
        """Retrieve an image and apply transformations."""
        image_path = self.record_list[idx]
        image = Image.open(image_path).convert('RGB')  # Ensure 3-channel RGB
        image = self.transform(image)
        return image

def collate_fn(batch):
    """Custom collate_fn function to load images and apply transformations."""
    # preprocess batch
    batch = [image for image in batch]
    return preprocess_train(batch)

def collate_fn_test(batch):
    """Custom collate_fn function to load images and apply transformations."""
    # preprocess batch
    batch = [image for image in batch]
    return preprocess_test(batch)

def build_dataloader(img_path, batch_size=32, num_workers=4, split='train'):
    
    if split == 'train':
        dataset = CustomDataset(img_path)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        return train_loader
    elif split == 'test':
        dataset = CustomDataset(img_path, transform=transforms.Compose([
            transforms.Resize((176, 176)),  # Resize to 176x176
            transforms.ToTensor(),  # Convert image to tensor
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (ImageNet stats)
        ]))
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_test)
        return test_loader

if __name__ == "__main__":
    img_path = 'resources/train.txt'
    ld = build_dataloader(img_path, batch_size=32, num_workers=4, split='test')
    sample  = next(iter(ld))
    print(len(sample))
    