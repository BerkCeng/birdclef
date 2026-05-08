import torch
import torch.nn as nn
import timm


class BirdCNN(nn.Module):
    def __init__(self, model_name='resnet34', num_classes=1, pretrained=True):
        super(BirdCNN, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = BirdCNN(model_name='efficientnet_b0', num_classes=5)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")
