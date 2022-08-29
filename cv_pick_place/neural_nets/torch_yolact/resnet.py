import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inchannels, channels, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = inchannels, out_channels = channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(num_features = channels)
        self.conv2 = nn.Conv2d(in_channels = channels,out_channels = channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(num_features = channels)
        self.conv3 = nn.Conv2d(in_channels = channels,out_channels = channels * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(num_features = channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # number of stages in resnet
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.inchannels = 64
        
        # C1 Stage of Resnet
        self.conv1 = nn.Conv2d(in_channels = 3, 
                                out_channels = 64, 
                                kernel_size=7, 
                                stride=2, 
                                padding=3, 
                                bias=False)
        self.bn1 = norm_layer(num_features = 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # C2 Stage of Resnet
        self._make_layer(block, channels = 64, blocks = layers[0])
        # C3 Stage of Resnet
        self._make_layer(block,channels = 128, blocks = layers[1], stride=2)
        # C4 Stage of Resnet
        self._make_layer(block,channels = 256, blocks = layers[2], stride=2)
        # C5 Stage of Resnet
        self._make_layer(block,channels = 512, blocks = layers[3], stride=2)

        # put Conv2d backbone layers into list
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        # Downsample if stride isn't default or if input channels aren't of proper size 
        if stride != 1 or self.inchannels != channels * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inchannels, channels * block.expansion,
                                                 kernel_size=1, stride=stride, bias=False),
                                       self.norm_layer(num_features = channels * block.expansion))
        # Create layers
        layers = [block(self.inchannels, channels, stride, downsample, self.norm_layer)]
        self.inchannels = channels * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inchannels, channels, norm_layer=self.norm_layer))

        layer = nn.Sequential(*layers)

        self.channels.append(channels * block.expansion)
        self.layers.append(layer)

    def forward(self, x):
        """ Returns a list of convouts for each layer. """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            outs.append(x)

        return tuple(outs)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)
        print(f'\nBackbone is initiated with {path}.\n')

if __name__ == '__main__':
    def construct_backbone(cfg_backbone=ResNet):
        # resnet101 has 3, 4, 23, 3 blocks for each stage
        # resnet50 has 3, 4, 6, 3 blocks for each stage
        backbone = cfg_backbone([3, 4, 23, 3])

        # Add downsampling layers until we reach the number we need
        selected_layers=[1, 2, 3]
        num_layers = max(selected_layers) + 1

        while len(backbone.layers) < num_layers:
            backbone.add_layer()

        return backbone

    from torchvision import models
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    model = construct_backbone().to(device)
    summary(model, (3, 550, 550))
    input=torch.randn(1, 3, 550, 550)
    backbone=construct_backbone()(input)

    print('backbone output features :', len(backbone))
    print('C2 output shape : ', backbone[0].shape)
    print('C3 output shape : ', backbone[1].shape)
    print('C4 output shape : ', backbone[2].shape)
    print('C5 output shape : ', backbone[3].shape)