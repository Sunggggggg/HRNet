import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets


def get_loaders(root, batch_size, resolution, num_workers=4):
    # root => data_dir = "./recaptcha-dataset/Large"

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = transforms.Compose([
                    transforms.Resize([resolution, resolution]),
                    transforms.RandomResizedCrop(resolution),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                    ])

    class_names = ['Bicycle', 'Bridge', 'Bus', 'Car', 
               'Chimney', 'Crosswalk', 'Hydrant', 
               'Motorcycle', 'Palm', 'Traffic Light']

    print("Initializing Datasets and Dataloaders...")

    image_datasets = datasets.ImageFolder(root, data_transforms)
    num_data = len(image_datasets)
    indices = np.arange(num_data)
    np.random.shuffle(indices)  

    train_size = int(num_data*0.8)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_set = Subset(image_datasets, train_indices)
    val_set = Subset(image_datasets, val_indices)

    print('Number of training data:', len(train_set))
    print('Number of validation data:', len(val_set))

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True, num_workers = num_workers)

    return train_loader, val_loader
