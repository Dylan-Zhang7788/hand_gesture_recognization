import torch
from torchvision import models
import torch.nn as nn

vgg16 = models.vgg16(pretrained=True)

vgg = vgg16.features
for param in vgg.parameters():
    param.requires_grad_(False)


class MyVggModel(nn.Module):
    def __init__(self):
        super(MyVggModel, self).__init__()
        # 预训练的vgg16的特征提取层
        self.vgg = vgg
        # 添加新的全连接层
        self.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 5),
            nn.Softmax(dim=1)
        )

    # 定义网络的向前传播路径

    def forward(self, x):
        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output
