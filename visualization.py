import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
import train_facial_expression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loss_graph(file_path):
    with open(f'{file_path}/loss.pickle', 'rb') as file:
        # Load the data from the pickle file
        data = pickle.load(file)

    plt.figure()
    plt.plot(data['Train loss'], label='Train loss')
    plt.plot(data['Val loss'], label='Val loss')
    plt.title('Loss')
    plt.grid()
    plt.legend()
    plt.show(block=False)
    plt.savefig(f'{file_path}/loss_graph.png')


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

def confusion_matrix(model, test_loader, file_path, checkpoint=None):
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

    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j, i, str(round(cm_normalized[i, j],2)), ha='center', va='center',
                    color='white' if cm_normalized[i, j] > np.max(cm_normalized) / 2 else 'black')
    plt.title(f'Test accuracy: {round(test_accuracy, 2)}')
    plt.show(block=False)
    plt.savefig(f'{file_path}/confusion matrix.png')


if __name__ == '__main__':
    dir = '2023_06_17_17_19_08'
    path = "./FER2013"
    train_loader, val_loader, test_loader, num_classes = train_facial_expression.prep_data(path)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, num_classes)  # Adjust the last fully connected layer for the correct number of classes
    confusion_matrix(model, test_loader, file_path=f'results/{dir}', checkpoint=f'./results/{dir}/checkpoint_validation_best.pth')
    loss_graph(f'results/{dir}')
