import torch
import torch.nn as nn
import timm


class BirdCRNN(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=1, hidden_size=128, num_layers=1, pretrained=True):
        super(BirdCRNN, self).__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')

        with torch.no_grad():
            dummy_x = torch.randn(1, 3, 224, 224)
            cnn_out = self.cnn(dummy_x)
            self.cnn_out_channels = cnn_out.shape[1]
            self.cnn_out_h = cnn_out.shape[2]
            self.input_size = self.cnn_out_channels * self.cnn_out_h

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.shape
        features = features.permute(0, 3, 1, 2).contiguous()
        features = features.view(b, w, c * h)

        lstm_out, _ = self.lstm(features)
        out = torch.mean(lstm_out, dim=1)
        logits = self.fc(out)
        return logits


if __name__ == "__main__":
    model = BirdCRNN(num_classes=5)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")
