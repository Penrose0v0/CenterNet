import torch
import torch.nn as nn
import torchvision.models as models
import math
from component import Decoder, Head

class CenterNet(nn.Module):
    def __init__(self, num_classes=20):
        super(CenterNet, self).__init__()
        # self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        # backbone_dict = torch.hub.load_state_dict_from_url(
        #     "https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth",
        #     model_dir="weights/backbone"
        # )
        # self.backbone.load_state_dict(backbone_dict)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone_dict = torch.hub.load_state_dict_from_url(
            "https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth",
            model_dir="weights/backbone"
        )
        self.backbone.load_state_dict(backbone_dict)
        # self.backbone = models.resnet18()
        # backbone_dict = torch.hub.load_state_dict_from_url(
        #     "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth",
        #     model_dir="weights/backbone"
        # )
        # self.backbone.load_state_dict(backbone_dict)

        self.backbone = torch.nn.Sequential(*self.backbone.children())[:-2]

        self.decoder = Decoder()
        self.head = Head(num_classes=num_classes)

        self.init_weight()

    def init_weight(self):
        for c in [self.decoder, self.head]:
            for m in c.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        self.head.hm_head[-2].weight.data.fill_(0)
        self.head.hm_head[-2].bias.data.fill_(-2.19)

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
