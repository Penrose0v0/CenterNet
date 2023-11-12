import torch
import torch.nn as nn
import torchvision.models as models
from component import Decoder, Head

class CenterNet(nn.Module):
    def __init__(self, num_classes=20):
        super(CenterNet, self).__init__()
        self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        # self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = torch.nn.Sequential(*self.backbone.children())[:-2]

        self.decoder = Decoder()
        self.head = Head(num_classes=num_classes)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)

        hm_pred, wh_pred, offset_pred = map(lambda y: y.permute(0, 2, 3, 1), self.head(x))
        return hm_pred, wh_pred, offset_pred

if __name__ == '__main__':
    net = CenterNet()
    sample = torch.rand(8, 3, 1024, 512)  # b * c * h * w
    sample = sample * 255
    hm, wh, offset = net(sample)
    print(hm.shape, wh.shape, offset.shape)
