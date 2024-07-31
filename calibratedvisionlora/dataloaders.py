
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split


class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]

        # With probability p = 0.5 change the label to the following one 
        if np.random.rand() < 0.5: 
            target = (target + 1) % self.num_classes
            
        return data, target

    
def filter_classes(dataset, target_classes):
    """
    Filter the dataset to include only samples from the target classes
    """
    targets = np.array(dataset._labels)
    mask = np.isin(targets, target_classes)
    indices = np.where(mask)[0]
    return Subset(dataset, indices)


def sample(dataset, size, batch_size): 
    subset_indices = torch.randperm(len(dataset))[:size]  
    dataset_subset = Subset(dataset, subset_indices)
    dataloader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False, num_workers=1)
    return dataloader


def create_food_data_loaders(transform, n_classes, batch_size, datasets_dir='../datasets', **kwargs): 

    train_size = kwargs.get('train_size', 1024)
    test_size = kwargs.get('test_size', 512)
    label_noise = kwargs.get('label_noise', True)
    
    # Download the dataset
    trainset = torchvision.datasets.Food101(root=datasets_dir, split='train', download=True, transform=transform)
    testset = torchvision.datasets.Food101(root=datasets_dir, split='test', download=True, transform=transform)

    # Keep only samples for the first n classes
    target_classes = list(range(n_classes))
    trainset_filtered = filter_classes(trainset, target_classes)
    testset_filtered = filter_classes(testset, target_classes)

    if label_noise:
        trainset_filtered = NoisyDataset(trainset_filtered, n_classes)
        testset_filtered = NoisyDataset(testset_filtered, n_classes)

    trainloader = DataLoader(trainset_filtered, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset_filtered, batch_size=batch_size, shuffle=False, num_workers=2)
    
    trainloader =sample(trainset_filtered, train_size, batch_size)
    testloader =sample(testset_filtered, test_size, batch_size)

    print(f'Training Samples - Train {len(trainloader) * batch_size} Test {len(testloader) * batch_size}')
    print(f'Batch size: {batch_size}')

    return trainloader, testloader