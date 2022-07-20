import torch.nn as nn


class ModelwithLinear(nn.Module):
    def __init__(self, model, inplanes, num_classes=10):
        super(ModelwithLinear, self).__init__()
        self.model = model
        
        self.classifier = nn.Linear(inplanes, num_classes)

    def forward(self, img):
        x = self.model(img)
        out = self.classifier(x)
        return out
    
class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, name='resnet50', feat_dim=512, num_classes=10):
        super(LinearClassifier, self).__init__()
        # _, feat_dim = model_dict[name]
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.classifier(features)
