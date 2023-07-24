import torch, os, random, argparse, time, copy
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from hrnet import HRNet
from loader import get_loaders

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_option():
    parser = argparse.ArgumentParser('HRNet train', add_help=False)
    parser.add_argument('--dataset_dir', type=str, help="dataset dir")
    parser.add_argument('--batch_size', type=int, help="batch size for single GPU")
    parser.add_argument('--epoch', type=int, help="training epoches")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--output', type=str, help="Run testdataset and eval")
    parser.add_argument('--resume', help='resume from checkpoint', default = None)

    args, unparsed = parser.parse_known_args()
    return args

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()     # Update
                        optimizer.step()    # optimizer Update

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

if __name__ == "__main__" :
    # Set Hyperparameters
    args = parse_option()

    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    epochs = args.epoch
    lr = args.lr
    output_name = args.output
    checkpoint = args.resume

    weights_path = output_name
    gamma = 0.7
    seed = 42
    seed_everything(seed)

    # Set dataLoader
    image_size = 128
    train_loader, valid_loader = get_loaders(dataset_dir, batch_size = batch_size, resolution = image_size)

    # Set model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if checkpoint :
        model = torch.load(checkpoint).to(device)
    else :
        model = HRNet(
            in_ch = 3,
            mid_ch = 16,
            out_ch = 512
        ).to(device)

    # train
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    dataloaders = {'train' : train_loader, 'val' : valid_loader}
    model, val_history = train_model(model, dataloaders, criterion, optimizer, num_epochs=25)
    
    torch.save(model, weights_path + '/final_model.pth')