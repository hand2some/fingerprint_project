"""
基于VCG16网络的OCT指纹防伪方法
The OCT Fingerprint antifake methods that based on VCG16 Network


"""
from torch.utils.data import random_split
from setting import nn, torch, F, test_environment, train_environment, Adam, lr_scheduler
from torch.nn import BCELoss
from dataload import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter


class VCT16(nn.Module):
	def __init__(self, in_channel=1):
		super(VCT16, self).__init__()
		self.conv_layer1 = self.new_conv(3, 1, 64)
		self.conv_layer2 = self.new_conv(3, 64, 128)
		self.conv_layer3 = self.new_conv(4, 128, 256)
		self.conv_layer4 = self.new_conv(4, 256, 512)
		self.conv_layer5 = self.new_conv(4, 512, 512)
		self.flatten = nn.Flatten(1)
		self.fc = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Linear(4096, 2)
		)

	def new_conv(self, layer_nums, in_channel, out_channel):
		conv_layers = []
		for i in range(layer_nums):
			tmp = [
				nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(out_channel),
			]
			in_channel = out_channel
			if i != layer_nums - 1:
				tmp.append(nn.ReLU())
				conv_layers.extend(tmp)
			else:
				tmp.append(nn.MaxPool2d(kernel_size=2, stride=2))
				conv_layers.extend(tmp)
		return nn.Sequential(*conv_layers)

	def forward(self, x):
		out = self.conv_layer1(x)
		out = self.conv_layer2(out)
		out = self.conv_layer3(out)
		out = self.conv_layer4(out)
		out = self.conv_layer5(out)
		out = self.flatten(out)
		out = self.fc(out)
		return out


if __name__ == '__main__':
	input_tensor = torch.randn((1, 1, 224, 224))
	vct16 = VCT16()
	output_tensor = vct16(input_tensor)
	print(output_tensor.shape)
