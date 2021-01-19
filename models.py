from torch import nn
from torchvision.models import resnet50
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.lstm1 = nn.LSTM(2048, 512)
        self.lstm2 = nn.LSTM(512, 512, batch_first=True)
        self.fc = nn.Linear(512, 2)

    def forward(self, inputs):
        xs = torch.zeros((inputs.size(0), inputs.size(1), 2048)).to(inputs.device)
        for i in range(inputs.size(1)):
            input = inputs[:, i]
            x = self.backbone.conv1(input)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            xs[:, i] = x
        x, (hn, cn) = self.lstm1(xs)
        x, (hn, cn) = self.lstm2(x)
        x = self.fc(x[:, -1, :])
        return x

if __name__ == '__main__':
    model = Model()
    model(torch.zeros((2, 3, 3, 256, 256)))
