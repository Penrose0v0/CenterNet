import torch
import torch.nn as nn
import torchvision.models as models
import math
from component import Decoder, Head

class CenterNet(nn.Module):
    def __init__(self, num_classes=1):
        super(CenterNet, self).__init__()
        self.backbone = models.resnet50()
        self.backbone = torch.nn.Sequential(*self.backbone.children())[:-2]

        self.decoder = Decoder()
        self.head = Head(num_classes=num_classes)

        # Initialize the parameter of each module
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # I do not know what the following code can do
        self.head.hm_head[-2].weight.data.fill_(0)
        self.head.hm_head[-2].bias.data.fill_(-2.19)

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)

        hm_pred, wh_pred, offset_pred = map(lambda y: y.permute(0, 2, 3, 1), self.head(x))
        return hm_pred, wh_pred, offset_pred

if __name__ == '__main__':
    net = CenterNet()
    sample = torch.rand(8, 3, 512, 512)
    sample = sample * 255
    hm, wh, offset = net(sample)
    print(hm.shape, wh.shape, offset.shape)
