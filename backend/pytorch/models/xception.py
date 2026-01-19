import torch.nn as nn
from torchvision import models

def get_xception(num_classes=2):
    model = models.mobilenet_v2(pretrained=True)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    return model
