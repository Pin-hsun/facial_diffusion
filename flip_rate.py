import glob
import os
import shutil

import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms

dataset = 'AffectNet'
classifier = 'smile'
assert dataset in ['CelebA', 'CelebAHQmask', 'AffectNet']
assert classifier in ['smile', 'happy', 'flip']
prj = 'test_affectnet_au_XY_condTrans_maskAU'

# INPUT_PATH = f'/home/gloria/projects/facial_diffusion/experiments/{prj}/results/test/0'

# OUTPUT_PATH = f'/home/glory/projects/Palette-Image-to-Image-Diffusion-Models/experiments/{prj}/compare'

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, layer=18):
        super(ResNetClassifier, self).__init__()
        # Load pretrained ResNet18 model
        assert layer in [18, 34, 50]
        if layer == 18:
            self.backbone = models.resnet18(pretrained=True)
        elif layer == 34:
            self.backbone = models.resnet34(pretrained=True)
        elif layer == 50:
            self.backbone = models.resnet50(pretrained=True)
        # Modify the fully connected layer to output 2 classes
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def flip_rate(dataset, image_paths):
    classifier_path = os.path.join('/mnt/nas/Data/pinhsun/logs/classifier', dataset, 'resnet34', 'checkpoints', 'classifier.pth')
    classifier = torch.load(classifier_path)
    classifier.eval()
    classifier.cuda()

    pred_ls = []
    for i in range(len(image_paths)):
        img = Image.open(image_paths[i])
        img = transforms.ToTensor()(img).unsqueeze(0).cuda()
        img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
        pred = classifier(img)
        pred = np.argmax(pred.squeeze().detach().cpu().numpy())
        pred_ls.append(pred)
    flip_rate = sum(pred_ls) / len(pred_ls)

    return flip_rate

def class_prediction_rate(dataset, classifier, image_paths):
    if classifier == 'smile':
        classifier_path = '/mnt/nas/Data/pinhsun/logs/classifier/CelebAHQmask/resnet34/checkpoints/classifier.pth'
    elif classifier == 'happy':
        classifier_path = '/mnt/nas/Data/pinhsun/logs/classifier/AffectNet/resnet34/checkpoints/classifier.pth'
    elif classifier == 'flip':
        classifier_path = os.path.join('/mnt/nas/Data/pinhsun/logs/classifier', dataset, 'resnet34', 'checkpoints',
                                       'classifier.pth')
    classifier = torch.load(classifier_path)
    classifier.eval()
    classifier.cuda()

    pred_ls = []
    for i in range(len(image_paths)):
        img = Image.open(image_paths[i])
        img = transforms.ToTensor()(img).unsqueeze(0).cuda()
        img = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(img)
        pred = classifier(img)
        pred = np.argmax(pred.squeeze().detach().cpu().numpy())
        pred_ls.append(pred)
    flip_rate = sum(pred_ls) / len(pred_ls)

    return flip_rate, pred_ls

if __name__ == '__main__':
    # image_paths = []
    # for dirpath, dirnames, filenames in os.walk(INPUT_PATH):
    #     for file in filenames:
    #         if file.endswith('.jpg') and file.startswith('Out'):
    #             image_filepath = os.path.join(dirpath, file)
    #             image_paths.append(image_filepath)

    INPUT_PATH = '/mnt/nas/Data/pinhsun/logs/GAN/AffetNet/Neutral/images'
    image_paths = glob.glob(INPUT_PATH + '/*')
    assert len(image_paths) > 0, "No image found in the input directory"
    print(len(image_paths))
    image_paths.sort()
    print(INPUT_PATH)
    affectnet_happy_rate, _ = class_prediction_rate(dataset, "happy", image_paths)
    print("affectnet_happy_rate: ", affectnet_happy_rate)
    affectnet_smile_rate, affectnet_smile_pred = class_prediction_rate(dataset, "smile", image_paths)
    print("affectnet_smile_rate: ", affectnet_smile_rate)
    for i in range(500):
        if affectnet_smile_pred[i] == 1:
            file_name = os.path.basename(image_paths[i])
            shutil.copy(image_paths[i], '/home/gloria/projects/NTU_Parkinson_Project/results/dataset_compare/neutral_smile/AffectNet/' + file_name)


    INPUT_PATH = '/mnt/nas/Data/pinhsun/logs/GAN/CelebA-HQ/Happiness/images'
    image_paths = glob.glob(INPUT_PATH + '/*')
    assert len(image_paths) > 0, "No image found in the input directory"
    print(len(image_paths))
    image_paths.sort()
    print(INPUT_PATH)
    celeba_smile_rate, _ = class_prediction_rate(dataset, "smile", image_paths)
    print("celeba_smile_rate: ", celeba_smile_rate)
    celeba_happy_rate, celeba_happy_pred = class_prediction_rate(dataset, "happy", image_paths)
    print("celeba_happy_rate: ", celeba_happy_rate)
    for i in range(500):
        if celeba_happy_pred[i] == 0:
            file_name = os.path.basename(image_paths[i])
            shutil.copy(image_paths[i], '/home/gloria/projects/NTU_Parkinson_Project/results/dataset_compare/neutral_smile/CelebAHQ/' + file_name)

