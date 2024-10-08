import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, dilation=1, relu=True, weight_decay=None):
        super(Conv2dLayer, self).__init__()

        padding = dilation if dilation > 1 else 1

        self.conv = nn.Conv2d(
            input_channels, 
            output_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation
        )

        # Custom weight initialization
        nn.init.normal_(self.conv.weight, std=5e-2)

        # Bias initialization
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = self.relu(x)
        return x

class ColorizeNet(nn.Module):
    def __init__(self, train=True, common_params=None, net_params=None):
        super(ColorizeNet, self).__init__()
        self.weight_decay = 0.0001
        self.batch_size = 8
        # Convolutional layers defined elegantly with nn.Sequential
        self.conv_block1 = nn.Sequential(
            Conv2dLayer(1, 64, weight_decay=self.weight_decay),
            Conv2dLayer(64, 64, stride=2, weight_decay=self.weight_decay),
            nn.BatchNorm2d(64)
        )

        self.conv_block2 = nn.Sequential(
            Conv2dLayer(64, 128, weight_decay=self.weight_decay),
            Conv2dLayer(128, 128, stride=2, weight_decay=self.weight_decay),
            nn.BatchNorm2d(128)
        )

        self.conv_block3 = nn.Sequential(
            Conv2dLayer(128, 256, weight_decay=self.weight_decay),
            Conv2dLayer(256, 256, weight_decay=self.weight_decay),
            Conv2dLayer(256, 256, stride=2, weight_decay=self.weight_decay),
            nn.BatchNorm2d(256)
        )

        self.conv_block4 = nn.Sequential(
            Conv2dLayer(256, 512, weight_decay=self.weight_decay),
            Conv2dLayer(512, 512, weight_decay=self.weight_decay),
            Conv2dLayer(512, 512, weight_decay=self.weight_decay),
            nn.BatchNorm2d(512)
        )

        self.dilated_block = nn.Sequential(
            Conv2dLayer(512, 512, dilation=2, weight_decay=self.weight_decay),
            Conv2dLayer(512, 512, dilation=2, weight_decay=self.weight_decay),
            Conv2dLayer(512, 512, dilation=2, weight_decay=self.weight_decay),
            nn.BatchNorm2d(512),
            Conv2dLayer(512, 512, dilation=2, weight_decay=self.weight_decay),
            Conv2dLayer(512, 512, dilation=2, weight_decay=self.weight_decay),
            Conv2dLayer(512, 512, dilation=2, weight_decay=self.weight_decay),
            nn.BatchNorm2d(512)
        )

        self.conv_block7 = nn.Sequential(
            Conv2dLayer(512, 512, weight_decay=self.weight_decay),
            Conv2dLayer(512, 512, weight_decay=self.weight_decay),
            Conv2dLayer(512, 512, weight_decay=self.weight_decay),
            nn.BatchNorm2d(512)
        )

        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256)
        )
        nn.init.normal_(self.deconv_block[0].weight, std=5e-2)  # Custom weight initialization for deconv

        self.final_block = nn.Sequential(
            Conv2dLayer(256, 256, weight_decay=self.weight_decay),
            Conv2dLayer(256, 256, weight_decay=self.weight_decay),
            Conv2dLayer(256, 313, kernel_size=1, relu=False, weight_decay=self.weight_decay)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.dilated_block(x)
        x = self.conv_block7(x)
        x = self.deconv_block(x)
        x = self.final_block(x)
        return x


class ColorizeLoss(nn.Module):
    def __init__(self, batch_size):
        super(ColorizeLoss, self).__init__()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions, prior_boost_nongray, ground_truth):
        flat_predictions = predictions.view(-1, 313)
        flat_ground_truth = ground_truth.view(-1, 313)
        
        # Calculate the initial loss
        initial_loss = self.criterion(flat_predictions, flat_ground_truth) / self.batch_size
        
        # Calculate gradients
        gradients = torch.autograd.grad(initial_loss, predictions, retain_graph=True)[0]
        
        # Compute the new loss
        new_loss = torch.sum(gradients * predictions * prior_boost_nongray) + initial_loss

        return new_loss, initial_loss



if __name__ == '__main__':
    net = ColorizeNet()
    print(net)
    x = torch.randn(1, 1, 224, 224)
    y = net(x)
    net.loss(y, torch.randn(1, 1, 224, 224), torch.randn(1, 313, 224, 224))
    print(y.size())