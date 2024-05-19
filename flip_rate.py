import os
import numpy as np
import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms

dataset = 'CelebA'
assert dataset in ['CelebA', 'CelebAHQmask', 'AffectNet']
prj = 'test_celeba_smile128_au_smile_conTrans'

INPUT_PATH = f'/home/glory/projects/Palette-Image-to-Image-Diffusion-Models/experiments/{prj}/results/test/0'
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

if __name__ == '__main__':
    image_paths = []
    for dirpath, dirnames, filenames in os.walk(INPUT_PATH):
        for file in filenames:
            if file.endswith('.jpg') and file.startswith('Out'):
                image_filepath = os.path.join(dirpath, file)
                image_paths.append(image_filepath)
    assert len(image_paths) > 0, "No image found in the input directory"
    print(len(image_paths))
    image_paths.sort()

    classifier_path = os.path.join('/media/ziyi/glory/logs_pin/classifier', dataset, 'resnet34', 'checkpoints', 'classifier.pth')
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
    print(prj, " flip_rate: ",flip_rate)
