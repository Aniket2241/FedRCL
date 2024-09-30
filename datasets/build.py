from utils.registry import Registry
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Normalize, CenterCrop
import yaml
import torch
import os

# Create a registry for datasets
DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
"""
# Register existing datasets
DATASET_REGISTRY.register(CIFAR10)
DATASET_REGISTRY.register(CIFAR100)
DATASET_REGISTRY.register(ImageFolder) 

# Register brain dataset using ImageFolder
@DATASET_REGISTRY.register()
def brain_dataset(root, train=True, transform=None):
    dataset_path = os.path.join(root, 'train' if train else 'test')
    return ImageFolder(root=dataset_path, transform=transform)

__all__ = ['build_dataset', 'build_datasets']

def get_transform(args, train, config):
    if 'brain_dataset' in args.dataset.name:
        # Define transformations specific to the brain dataset
        normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        imsize = config['imsize']  # Get image size from config
        if train:
            transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(imsize, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                CenterCrop(imsize),
                ToTensor(),
                normalize
            ])
    else:
        # Other datasets (CIFAR, etc.)
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        normalize = Normalize(config['mean'], config['std'])
        imsize = config['imsize']
        if train:
            transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomCrop(imsize, padding=4),
                transforms.RandomHorizontalFlip(),
                ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                CenterCrop(imsize),
                ToTensor(),
                normalize
            ])
    
    return transform


def build_dataset(args, train=True):
    if args.verbose and train:
        print(DATASET_REGISTRY)

    download = args.dataset.download if args.dataset.get('download') else False

    with open('/content/FedRCL/datasets/configs.yaml', 'r') as f:
        dataset_config = yaml.safe_load(f)[args.dataset.name]
    
    transform = get_transform(args, train, dataset_config)
    
    # Check if the dataset is the brain dataset
    if 'brain_dataset' in args.dataset.name:
        dataset = brain_dataset(root=args.dataset.path, train=train, transform=transform)
    else:
        dataset = DATASET_REGISTRY.get(args.dataset.name)(
            root=args.dataset.path, download=download, train=train, transform=transform
        ) if len(args.dataset.path) > 0 else None

    return dataset

def build_datasets(args):
    train_dataset = build_dataset(args, train=True)
    test_dataset = build_dataset(args, train=False)
    
    datasets = {
        "train": train_dataset,
        "test": test_dataset,
    }

    return datasets