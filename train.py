import torch, os, random, argparse
import numpy as np

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
from hrnet import HRNet
from loader import get_loaders

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

def train(model, device, train_loader, valid_loader, weights_path = './Model_weight', epochs = 20, lr = 0.0001, gamma = 0.7):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    os.makedirs(weights_path, exist_ok = True)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0

        for data, label in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(device)
                label = label.to(device)

                val_output = model(data)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )

        if (epoch + 1) % 10 == 0 :
            torch.save(model, weights_path + f'/HRNet_{epoch + 1}.pth')
    
    return model

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
    image_size = 224
    train_loader, valid_loader = get_loaders(dataset_dir, image_size = image_size, batch_size = batch_size)

    # Set model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if checkpoint :
        model = torch.load(checkpoint).to(device)
    else :
        model = HRNet(
            in_ch = 3,
            mid_ch = 16,
            out_ch = 10
        ).to(device)

    # train
    train_model = train(model, device, train_loader, valid_loader, weights_path, epochs, lr, gamma) 
    torch.save(train_model, weights_path + '/final_model.pth')