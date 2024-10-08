import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import preprocess

class CustomDataset(Dataset):
    """Custom Dataset for image loading and processing using PyTorch operations."""

    def __init__(self, img_path,  transform=None):
        self.img_path = img_path
        self.image_size = 224
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize (ImageNet stats)
        ])

    def __getitem__(self, idx):
        """Retrieve an image and apply transformations."""
        image_path = self.record_list[idx]
        image = Image.open(image_path).convert('RGB')  # Ensure 3-channel RGB
        return self.transform(image)

def collate_fn(batch):
    """Custom collate_fn function to load images and apply transformations."""
    # preprocess batch
    batch = [image for image in batch]
    return preprocess(batch)

if __name__ == "__main__":
    img_path = 'resources/train.txt'
    dataset = CustomDataset(img_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for data in train_loader:
        data_l, gt_ab_313, prior_boost_nongray = data
        print(data_l.shape, gt_ab_313.shape, prior_boost_nongray.shape)
        break