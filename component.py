import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, in_channels=2048, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.bn_momentum = bn_momentum
        self.deconv_with_bias = False

        self.deconv_layers = self.make_deconv_layer(
            num_layers=3,
            num_filters=[256, 128, 64],
            num_kernels=[4, 4, 4],
        )

    def make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            num_filter = num_filters[i]

            fc = nn.Conv2d(in_channels=self.in_channels,
                           out_channels=num_filter,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           dilation=1,
                           bias=True)

            up = nn.ConvTranspose2d(
                in_channels=num_filter,
                out_channels=num_filter,
                kernel_size=kernel,
                stride=2,
                padding=1,
                output_padding=0,
                bias=self.deconv_with_bias)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(num_filter, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(num_filter, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = num_filter

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)


class Head(nn.Module):
    def __init__(self, num_classes=1, channel=64, bn_momentum=0.1):
        super(Head, self).__init__()

        # heatmap
        self.hm_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.hm_head[-2].bias.data.fill_(-2.19)

        # bounding boxes height and width
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )
        for m in self.wh_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # center point offset
        self.offset_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )
        for m in self.offset_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        hm = self.hm_head(x)
        wh = self.wh_head(x)
        offset = self.offset_head(x)

        return hm, wh, offset
