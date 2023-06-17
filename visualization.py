import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
import train_facial_expression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_graph(file_path):
    with open(file_path, 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)

    plt.figure()
    plt.plot(data['Train loss'], label='Train loss')
    plt.plot(data['Val loss'], label='Val loss')
    plt.title('Loss')
    plt.grid()
    plt.legend()
    plt.show()


def calculate_accuracy(model, dataloader, device):
    model.eval() # put in evaluation mode,  turn of DropOut, BatchNorm uses learned statistics
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([7,7], int)
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix

def confusion_matrix(model, test_loader, checkpoint=None):
    if checkpoint is not None:
        model = model.to(device)
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state['model_state_dict'])

    test_accuracy, confusion_matrix = calculate_accuracy(model, test_loader, device)
    print("test accuracy: {:.3f}%".format(test_accuracy))
    classes = np.array(['angry', 'disgust', 'fear', 'happy', 'netural', 'sad', 'surprise'])
    # plot confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=1000, cmap=plt.get_cmap('Blues'))
    plt.ylabel('Actual Category')
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Predicted Category')
    plt.xticks(range(len(classes)), classes)
    plt.show()

if __name__ == '__main__':
    path = "./FER2013"
    train_loader, val_loader, test_loader, num_classes = train_facial_expression.prep_data(path)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)  # Adjust the last fully connected layer for the correct number of classes
    confusion_matrix(model, test_loader, checkpoint='./dump_loss/checkpoint_last_2023_06_17_13_32_41.pth')