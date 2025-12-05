import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class EfficientNetLSTM(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, num_layers=1, bidirectional=False):
        super().__init__()
        base_model = efficientnet_b0(pretrained=True)

        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        feature_dim = base_model.classifier[1].in_features

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(lstm_output, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        seq = x.unsqueeze(1)
        lstm_out, (h_n, c_n) = self.lstm(seq)
        last_hidden = h_n[-1]
        logits = self.classifier(last_hidden)
        return logits
