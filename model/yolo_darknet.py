import torch
from torch import nn
from model import darknet19
import numpy as np

class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x

class YOLOV2(nn.Module):

    def __init__(self, num_classes=20, darknet = None,
                 anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
                          (11.2364, 10.0071)], ):
        super(YOLOV2, self).__init__()

        self.anchors = anchors

        self.num_classes = num_classes

        # self.backbone_input = darknet.input_conv
        # self.backbone_layer1 = darknet.layer1
        # self.backbone_layer2 = darknet.layer2
        # self.backbone_layer3 = darknet.layer3
        # self.backbone_layer4 = darknet.layer4
        self.backbone_conv1 = nn.Sequential(darknet.input_conv, darknet.layer1,
                                   darknet.layer2, darknet.layer3, darknet.layer4)

        self.backbone_resconv = darknet.layer5

        self.yolo_conv1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),),
            nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True), ),
        )

        self.yolo_conv2 =  nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.yolo_conv3 =  nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, (5 + self.num_classes) * len(anchors), kernel_size=1)
        )

        self.reorg = ReorgLayer()

    def forward(self, x):
        x = self.backbone_conv1(x)

        s = self.reorg(self.yolo_conv2(x))

        x = self.backbone_resconv(x)
        x = self.yolo_conv1(x)
        x = torch.cat([s, x], dim=1)
        x = self.yolo_conv3(x)

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
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
        # ref : https://github.com/tztztztztz/yolov2.pytorch
        self.start = 0
        fp = open(weights_file, 'rb')
        self.buf = np.fromfile(fp, dtype=np.float32)
        fp.close()
        self.dfs(model)

if __name__ == '__main__':
    darknet = darknet19.DarkNet19(num_classes=20)
    model = YOLOV2(darknet=darknet).cuda()
    model.load_weight(model, 'yolo-voc.weights')
    print(model)
    image = torch.randn([1, 3, 416, 416]).cuda()
    print(model(image).size())