import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np
import torch.nn as nn
from torchvision import datasets, models, transforms
import train_facial_expression
from PIL import Image
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_accuracy(model_original, model_cartoon, dataloader_original, dataloader_cartoon, device):
    model_original.eval() # put in evaluation mode,  turn of DropOut, BatchNorm uses learned statistics
    model_cartoon.eval()  # put in evaluation mode,  turn of DropOut, BatchNorm uses learned statistics
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([7,7], int)
    with torch.no_grad():
        for data_original, data_cartoon in zip(dataloader_original, dataloader_cartoon):
            images_original, labels = data_original
            images_original = images_original.to(device)
            labels = labels.to(device)
            outputs_original = model_original(images_original)
            outputs_original = F.softmax(outputs_original, dim=1)
            images_cartoon, _ = data_cartoon
            images_cartoon = images_cartoon.to(device)
            outputs_cartoon = model_cartoon(images_cartoon)
            outputs_cartoon = F.softmax(outputs_cartoon, dim=1)
            #outputs = torch.max(outputs_original, outputs_cartoon)
            outputs = outputs_original + outputs_cartoon
            _, predicted = torch.max(outputs, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix

def confusion_matrix(model_original, model_cartoon, test_loader_original, test_loader_cartoon, output_path,
                     checkpoint_orginal=None, checkpoint_cartoon=None):
    if checkpoint_orginal is not None:
        model_original = model_original.to(device)
        state = torch.load(checkpoint_orginal, map_location=device)
        model_original.load_state_dict(state['model_state_dict'])
    if checkpoint_cartoon is not None:
        model_cartoon = model_cartoon.to(device)
        state = torch.load(checkpoint_cartoon, map_location=device)
        model_cartoon.load_state_dict(state['model_state_dict'])

    test_accuracy, confusion_matrix = calculate_accuracy(model_original, model_cartoon, test_loader_original, test_loader_cartoon, device)
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
    #plt.show(block=False)
    plt.savefig(f'{output_path}/confusion matrix.png')


def img_shape():
    image = Image.open('./FER2013/test/angry/PrivateTest_88305.jpg')
    print(image.size)

if __name__ == '__main__':
    test_name = 'best_original_best_cartoon_sum_11'
    output_path = f'results_2_models/{test_name}'
    os.mkdir(output_path)
    checkpoint_orginal = f'./results/2023_06_18_20_48_52_optimizer_adam_init_lr_0.0001_cartoon_prec_0.0_chunk2/checkpoint_validation_best.pth'
    checkpoint_cartoon = f'./results/2023_06_18_23_32_51_optimizer_adam_init_lr_0.0001_cartoon_prec_0.8_chunk2/checkpoint_validation_best.pth'
    path = "./FER2013"
    _, _, test_loader_original, num_classes = train_facial_expression.prep_data(path, test_mode="regular")
    _, _, test_loader_cartoon, num_classes = train_facial_expression.prep_data(path, test_mode="cartoon")
    model_original = models.resnet18(pretrained=True)
    model_cartoon = models.resnet18(pretrained=True)
    model_original.fc = nn.Linear(512, num_classes)  # Adjust the last fully connected layer for the correct number of classes
    model_cartoon.fc = nn.Linear(512, num_classes)  # Adjust the last fully connected layer for the correct number of classes
    confusion_matrix(model_original, model_cartoon, test_loader_original, test_loader_cartoon, output_path=output_path,
                     checkpoint_orginal=checkpoint_orginal, checkpoint_cartoon=checkpoint_cartoon)
