import torchvision.models as models
import torch.nn as nn

# Architettura utilizzata
def get_resnet_model(num_classes):
    model = models.resnet18(weights='DEFAULT')  
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
