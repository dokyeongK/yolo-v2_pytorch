from torch import nn
import torch
import numpy as np


class DarkNet19(nn.Module):
    def __init__(self, num_classes=80, num_bboxes=2):
        super(DarkNet19, self).__init__()

        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        # 6개의 layer.

        # [32]
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # [Maxpool, 64]
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # [Maxpool, 128, 64, 128]
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # [Maxpool, 256, 128, 256]
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # [Maxpool, 512, 256, 512, 256, 512]
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # [Maxpool, 1024, 512, 1024, 512, 1024]
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # [Classifier > 1024, 1000]
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1000, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1000),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.softmax = nn.Softmax(dim=1)

        self.init_weight(self.input_conv)
        self.init_weight(self.layer1)
        self.init_weight(self.layer2)
        self.init_weight(self.layer3)
        self.init_weight(self.layer4)
        self.init_weight(self.layer5)
        # self.init_weight(self.classifier)

    def forward(self, x):

        x = self.input_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x) # [3, 1024, 13, 13]
        x = self.classifier(x)
        x = self.softmax(x)
        return x

    def init_weight(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_conv_bn(self, conv_model, bn_model):
        num_w = conv_model.weight.numel()
        num_b = bn_model.bias.numel()

        bn_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        bn_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        bn_model.running_mean.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        bn_model.running_var.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), bn_model.bias.size()))
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        self.start = self.start + num_w

    def load_conv(self, conv_model):
        num_w = conv_model.weight.numel()
        num_b = conv_model.bias.numel()
        conv_model.bias.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_b]), conv_model.bias.size()))
        self.start = self.start + num_b
        conv_model.weight.data.copy_(
            torch.reshape(torch.from_numpy(self.buf[self.start:self.start + num_w]), conv_model.weight.size()))
        self.start = self.start + num_w

    def dfs(self, m):
        children = list(m.children())
        for i, c in enumerate(children):
            if isinstance(c, torch.nn.Sequential):
                self.dfs(c)
            elif isinstance(c, torch.nn.Conv2d):
                if c.bias is not None:
                    self.load_conv(c)
                else:
                    self.load_conv_bn(c, children[i + 1])

    def load_weight(self, model, weights_file):
        self.start = 0
        fp = open(weights_file, 'rb')
        self.buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        self.dfs(model)

# if __name__ == '__main__':
#     model = DarkNet19(num_classes=3).cuda()
#     model.load_weight(model, weights_file='yolo-voc.weights')
#     print(model)
#     image = torch.randn([3, 3, 416, 416]).cuda()
#     print(model(image).size())