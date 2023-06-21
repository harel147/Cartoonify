import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import pickle
import time
import visualization
import numpy as np
import os
import random
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(123)

parser = argparse.ArgumentParser(description='face recognition resnet-18')
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--optimizer', default="adam", type=str, help='[sgd, adam]')
parser.add_argument('--scheduler', default="reduce", type=str, help='[reduce, cos]')
parser.add_argument('--lr_sgd', default=0.1, type=float)
parser.add_argument('--lr_adam', default=0.001, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--data_path', default='./FER2013', type=str)
parser.add_argument('--cartoon_prec', default=0.5, type=float)
parser.add_argument('--test_mode', default="regular", type=str, help='[regular, cartoon]')
parser.add_argument('--train_on_united', default="no", type=str, help='[no, yes]')

# Define the custom dataset class
class AugmentedDataset(Dataset):
    def __init__(self, original_folder, augmented_folder, transform=None, augment_prec=0.0):
        super(AugmentedDataset, self).__init__()
        self.original_dataset = datasets.ImageFolder(original_folder, transform=transform)
        self.augmented_dataset = datasets.ImageFolder(augmented_folder, transform=transform)
        self.transform = transform
        self.original_prec = 1 - augment_prec  # chance of using the original image

    def __getitem__(self, index):
        # Randomly decide whether to use original or augmented image
        use_original = random.random() < self.original_prec  # chance of using the original image
        if use_original:
            image, label = self.original_dataset[index]
        else:
            image, label = self.augmented_dataset[index]
        return image, label

    def __len__(self):
        return max(len(self.original_dataset), len(self.augmented_dataset))

    @property
    def classes(self):
        # Return the classes from the original dataset
        return self.original_dataset.classes


def evalute(model, loader, criterion):
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    total_loss = running_loss / len(loader)
    accuracy = correct / total
    return accuracy, total_loss


def train(args, model, optimizer, scheduler, criterion, num_epochs, train_loader, val_loader, folder_name):
    # Training loop
    train_loss_list = []
    val_loss_list = []
    best_val = np.inf
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss_list.append(running_loss / len(train_loader))

        val_acc, val_loss = evalute(model, val_loader, criterion)
        val_loss_list.append(val_loss)

        if args.optimizer == 'sgd':
            if args.scheduler == 'cos':
                scheduler.step()
            elif args.scheduler == 'reduce':
                scheduler.step(val_acc)

        if val_loss < best_val:
            best_val = val_loss
            # Save the checkpoint to a file
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / len(train_loader),
            }, f'./results/{folder_name}/checkpoint_validation_best.pth')

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train_Loss: {running_loss / len(train_loader):.4f}, Val_Loss: {val_loss:.4f}, Val_acc: {val_acc:.4f}")

        if epoch % 5 == 0:
            # Save the checkpoint to a file
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / len(train_loader),
            }, f'./results/{folder_name}/checkpoint_run.pth')
            data = {'Train loss': train_loss_list, 'Val loss': val_loss_list}
            with open(f'./results/{folder_name}/loss.pickle', 'wb') as file:
                # Dump the data into the pickle file
                pickle.dump(data, file)

    # Save the checkpoint to a file
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss / len(train_loader),
    }, f'./results/{folder_name}/checkpoint_last.pth')
    return train_loss_list, val_loss_list, model


def prep_data(path, cartoon_prec=0.5, test_mode="regular", batch_size=64, train_on_united="no"):
    # Define data transformations
    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        #transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # transforms.Lambda(lambda tensors: torch.stack(
        #     [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    ])

    # Load FER2013 dataset
    if train_on_united == "no":
        train_data = AugmentedDataset(f"{path}/train", f"{path}/train_cartoon", transform=transform, augment_prec=cartoon_prec)
    elif train_on_united == "yes":
        train_data = AugmentedDataset(f"{path}/train_united", f"{path}/train_cartoon", transform=transform, augment_prec=0.0)
    val_data = datasets.ImageFolder(f"{path}/validation", transform=transform)  # we took 10% from the original train set
    if test_mode == "regular":
        test_data = datasets.ImageFolder(f"{path}/test", transform=transform)
    elif test_mode == "cartoon":
        test_data = datasets.ImageFolder(f"{path}/test_cartoon", transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    num_classes  = len(train_data.classes)
    print(f"train data size: {len(train_data)}")
    print(f"val data size: {len(val_data)}")
    print(f"test data size: {len(test_data)}")
    print(f"num classes: {num_classes}")
    return train_loader, val_loader, test_loader, num_classes


def objective(trial):
    args = parser.parse_args()
    # Load pre-trained ResNet-18 model
    model = models.resnet18()
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
    model.fc = nn.Linear(512, 7)  # Adjust the last fully connected layer for the correct number of classes
    # Generate the model.
    model = model.to(device)
    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)  # log=True, will use log scale to interplolate between lr
    batch = trial.suggest_int("batch", 2, 128)  # log=True, will use log scale to interplolate between lr
    cartoon_prec = trial.suggest_float("cartoon_augmentation", 0, 1)  # log=True, will use log scale to interplolate between lr
    optimizer = getattr(optim, "Adam")(model.parameters(), lr=lr)
    print(f"lr: {lr}, batch size: {batch}, cartoon prec: {cartoon_prec}")
    train_loader, val_loader, test_loader, num_classes = prep_data(args.data_path, cartoon_prec, args.test_mode,
                                                                   batch, args.train_on_united)
    # Define loss function and optimizer and Hyper parameters
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        print(f"lr: {lr}, batch size: {batch}, cartoon prec: {cartoon_prec}")
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc, val_loss = evalute(model, val_loader, criterion)
        print(f"val_acc: {val_acc}")

        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing (accuracy)
        trial.report(val_acc, epoch)

        # then, Optuna can decide if the trial should be pruned
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


def optuna_main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=30)

    best_trial = study.best_trial
    print("Best trial:")
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

def main():
    args = parser.parse_args()
    start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    if args.optimizer=="adam":
        save_lr = args.lr_adam
    elif args.optimizer=="sgd":
        save_lr = args.lr_sgd
    folder_name = f'{start_time}_optimizer_{args.optimizer}_init_lr_{save_lr}_cartoon_prec_{args.cartoon_prec}'
    os.mkdir(f"./results/{folder_name}")

    train_loader, val_loader, test_loader, num_classes = prep_data(args.data_path, args.cartoon_prec, args.test_mode,
                                                                   args.batch_size, args.train_on_united)

    # Load pre-trained ResNet-18 model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)  # Adjust the last fully connected layer for the correct number of classes

    # Move model to device
    model = model.to(device)

    # Define loss function and optimizer and Hyper parameters
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        scheduler = None
        optimizer = optim.Adam(model.parameters(), lr=args.lr_adam)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(
        ), lr=args.lr_sgd, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        if args.scheduler == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs)
        elif args.scheduler == 'reduce':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.75, patience=5, verbose=True)

    train_loss_list, val_loss_list, model = train(args, model, optimizer, scheduler, criterion, args.epochs, train_loader, val_loader, folder_name)

    data = {'Train loss': train_loss_list, 'Val loss': val_loss_list}
    with open(f'./results/{folder_name}/loss.pickle', 'wb') as file:
        # Dump the data into the pickle file
        pickle.dump(data, file)

    visualization.loss_graph(f'./results/{folder_name}')

    # Evaluation
    # test_acc, test_loss = evaluate(model, test_loader, criterion)
    # print(f"Test Accuracy: {test_acc:.4f}")
    visualization.confusion_matrix(model, test_loader, file_path=f'./results/{folder_name}')

if __name__ == '__main__':
    # main()
    optuna_main()