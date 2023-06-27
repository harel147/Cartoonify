import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np
import torch.nn as nn
from torchvision import models
import train_facial_expression
import torch.nn.functional as F
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='test 2 models')
parser.add_argument('--test_name', default="best_original_best_cartoon_sum_1", type=str)
parser.add_argument('--model_checkpoint_original_testset', default="./results/2023_06_18_20_48_52_optimizer_adam_init_lr_0.0001_cartoon_prec_0.0_chunk2/checkpoint_validation_best.pth", type=str)
parser.add_argument('--model_checkpoint_cartoon_testset', default="./results/2023_06_18_23_32_51_optimizer_adam_init_lr_0.0001_cartoon_prec_0.8_chunk2/checkpoint_validation_best.pth", type=str)


def calculate_accuracy(model_original, model_cartoon, dataloader_original, dataloader_cartoon, device):
    """
    calculate combined accuracy of using two models, usually on test set
    """
    model_original.eval()  # put in evaluation mode,  turn of DropOut, BatchNorm uses learned statistics
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
    """
    plot and save model confusion matrix, usually on test set
    """
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


def main():
    args = parser.parse_args()
    output_path = f'results_2_models/{args.test_name}'
    os.mkdir(output_path)
    path = "./FER2013"
    _, _, test_loader_original, num_classes = train_facial_expression.prep_data(path, test_mode="regular")
    _, _, test_loader_cartoon, num_classes = train_facial_expression.prep_data(path, test_mode="cartoon")
    model_original = models.resnet18(pretrained=True)
    model_cartoon = models.resnet18(pretrained=True)
    model_original.fc = nn.Linear(512, num_classes)  # Adjust the last fully connected layer for the correct number of classes
    model_cartoon.fc = nn.Linear(512, num_classes)  # Adjust the last fully connected layer for the correct number of classes
    confusion_matrix(model_original, model_cartoon, test_loader_original, test_loader_cartoon, output_path=output_path,
                     checkpoint_orginal=args.model_checkpoint_original_testset, checkpoint_cartoon=args.model_checkpoint_cartoon_testset)

if __name__ == '__main__':
    main()