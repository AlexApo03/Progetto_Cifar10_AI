import pickle
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data
from PIL import Image

class Cifar10Batch(torch.utils.data.Dataset):
    def __init__(self, batch_files, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform
        print("Entering init...")
        
        for batch_file in batch_files:
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                self.data.append(batch['data'])
                self.labels.append(batch['labels'])

        # Concatenate all batches
        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_flat = self.data[idx].astype(np.uint8)
        img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)
        label = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)

        return img, label

def get_dataloaders(batch_size, train_batch_files=None, test_batch_file=None):
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Create datasets
    train_dataset = Cifar10Batch(batch_files=train_batch_files, transform=transform_train)
    test_dataset = Cifar10Batch(batch_files=[test_batch_file], transform=transform_test)

    # Create DataLoader
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader
