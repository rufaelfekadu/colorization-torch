import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from skimage import color
from skimage.transform import resize

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


import torch
from torch import nn

class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()

		self.l_cent = 50.
		self.l_norm = 100.
		self.ab_norm = 110.

	def normalize_l(self, in_l):
		return (in_l-self.l_cent)/self.l_norm

	def unnormalize_l(self, in_l):
		return in_l*self.l_norm + self.l_cent

	def normalize_ab(self, in_ab):
		return in_ab/self.ab_norm

	def unnormalize_ab(self, in_ab):
		return in_ab*self.ab_norm
    
class ECCVGenerator(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ECCVGenerator, self).__init__()
 
        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]
 
        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]
 
        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]
 
        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]
 
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]
 
        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]
 
        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]
 
        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
 
        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]
 
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)
 
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
 
    def forward(self, input_l):
        input_l = input_l.permute(0, 3, 1, 2)
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        # out_reg = self.model_out(self.softmax(conv8_3))
 
        return conv8_3.permute(0, 2, 3, 1)
    
    @staticmethod 
    def decode(data_l, conv8_313, rebalance=1):
        """
        Args:
            data_l   : [1, height, width, 1]
            conv8_313: [1, height/4, width/4, 313]
        Returns:
            img_rgb  : [height, width, 3]
        """
        data_l = data_l + 50
        _, height, width, _ = data_l.shape
        data_l = data_l[0, :, :, :]
        conv8_313 = conv8_313[0, :, :, :]
        enc_dir = './resources'
        conv8_313_rh = conv8_313 * rebalance
        class8_313_rh = softmax(conv8_313_rh)

        cc = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))
        
        data_ab = np.dot(class8_313_rh, cc)
        data_ab = resize(data_ab, (height, width))
        img_lab = np.concatenate((data_l, data_ab), axis=-1)
        img_rgb = color.lab2rgb(img_lab)

        return img_rgb

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1) # only difference


def eccv16(pretrained=True):
	model = ECCVGenerator()
	if(pretrained):
		import torch.utils.model_zoo as model_zoo
		model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth',map_location='cpu',check_hash=True))
	return model

class ColorizeNet(nn.Module):
    def __init__(self, ):
        super(ColorizeNet, self).__init__()
        self.weight_decay = 0.0001
        self.batch_size = 176
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
        # input: (batch_size, 256, 256, 1)
        x = x.permute(0, 3, 1, 2)  # (batch_size, 1, 256, 256)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.dilated_block(x)
        x = self.conv_block7(x)
        x = self.deconv_block(x)
        x = self.final_block(x)
        return x.permute(0, 2, 3, 1)  # (batch_size, 58, 58, 313)


class ColorizeLoss(nn.Module):
    def __init__(self, batch_size):
        super(ColorizeLoss, self).__init__()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predictions, prior_boost_nongray, ground_truth):
        flat_predictions = predictions.reshape(-1, 313)
        flat_ground_truth = ground_truth.reshape(-1, 313)
        # print(flat_predictions.shape, flat_ground_truth.shape, flat_prior.shape)
        # Calculate the initial loss
        initial_loss = self.criterion(flat_predictions, flat_ground_truth) / self.batch_size
        
        # Calculate gradients
        gradients = torch.autograd.grad(initial_loss, predictions, retain_graph=True)[0]
        
        # Compute the new loss
        new_loss = torch.sum(gradients * predictions * prior_boost_nongray) + initial_loss

        return new_loss, initial_loss



if __name__ == '__main__':
    net = ColorizeNet()
    criterion = ColorizeLoss(8)
    print(net)
    x = torch.randn(8, 224, 224, 1)
    y = net(x)
    print(y.shape)
    new_loss, inital_loss = criterion(y, torch.randn(8, 58, 58, 1), torch.randn(8,  58, 58, 313))
    print(new_loss, inital_loss)
