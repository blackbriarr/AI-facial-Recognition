import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

class LFWCW2Dataset(Dataset):
    def __init__(self, split_file, root="data", transform=None):
        self.root = root
        self.transform = transform or transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        
        # Load split
        with open(split_file, 'r') as f:
            self.data = [line.strip().split() for line in f.readlines()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path, label_str = self.data[idx]
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        label = int(label_str)
        if self.transform:
            image = self.transform(image)
        return image, label

# Usage:
# train_ds = LFWCW2Dataset("data/splits/lfw_cw2_train.txt")
# val_ds = LFWCW2Dataset("data/splits/lfw_cw2_val.txt")
# test_ds = LFWCW2Dataset("data/splits/lfw_cw2_test.txt")
