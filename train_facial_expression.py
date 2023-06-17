import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, random_split
import pickle
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def train(model, optimizer, criterion, num_epochs, train_loader, val_loader):
    # Training loop
    train_loss_list = []
    val_loss_list = []
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

        _, val_loss = evalute(model, val_loader, criterion)
        val_loss_list.append(val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train_Loss: {running_loss / len(train_loader):.4f}, Val_Loss: {val_loss:.4f}")

    return train_loss_list, val_loss_list, model


def prep_data(path):
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load FER2013 dataset
    train_data = datasets.ImageFolder(f"{path}/train", transform=transform)
    val_data = datasets.ImageFolder(f"{path}/validation", transform=transform)  # we took 10% from the original train set
    test_data = datasets.ImageFolder(f"{path}/test", transform=transform)

    # Create data loaders
    batch_size = 64

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    num_classes  = len(train_data.classes)
    print(f"train data size: {len(train_data)}")
    print(f"val data size: {len(val_data)}")
    print(f"test data size: {len(test_data)}")
    print(f"num classes: {num_classes}")
    return train_loader, val_loader, test_loader, num_classes


def main():
    path = "./FER2013"
    train_loader, val_loader, test_loader, num_classes = prep_data(path)

    # Load pre-trained ResNet-18 model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)  # Adjust the last fully connected layer for the correct number of classes

    # Move model to device
    model = model.to(device)

    # Define loss function and optimizer and Hyper parameters
    num_epochs = 2
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_list, val_loss_list, model = train(model, optimizer, criterion, num_epochs, train_loader, val_loader)

    data = {'Train loss': train_loss_list, 'Val loss': val_loss_list}
    with open(f'./dump_loss/loss_{time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())}.pickle', 'wb') as file:
        # Dump the data into the pickle file
        pickle.dump(data, file)

    # Evaluation
    test_acc, test_loss = evalute(model, test_loader, criterion)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()