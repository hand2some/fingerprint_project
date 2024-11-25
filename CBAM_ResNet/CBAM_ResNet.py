"""
基于Res-50网络的OCT指纹防伪方法
The OCT Fingerprint antifake methods that based on Res-50 Network
You will learn about the achievement about:
(1) Channel Attention
(2) Space Attention
(3) CBAM Model
(4) ResNet50 + CBAM
"""
from torch.utils.data import random_split

from setting import nn, torch, F, test_environment, train_environment, Adam, lr_scheduler
from torch.nn import BCELoss
from dataload import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter


class Channel_Attention(nn.Module):
	def __init__(self, in_channel):
		super(Channel_Attention, self).__init__()
		self.globalMaxPool = nn.AdaptiveMaxPool2d(1)
		self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
		self.mlp = nn.Sequential(
			nn.Conv2d(in_channel, in_channel // 16, kernel_size=1, bias=False),
			nn.ReLU(),
			nn.Conv2d(in_channel // 16, in_channel, kernel_size=1, bias=False)
		)

	def forward(self, x):
		max_pool_x = F.relu(self.globalMaxPool(x))
		avg_pool_x = F.relu(self.globalAvgPool(x))
		# MLP处理
		max_pool_x = self.mlp(max_pool_x)
		avg_pool_x = self.mlp(avg_pool_x)

		pool_x = max_pool_x + avg_pool_x

		return F.sigmoid(pool_x)


class Space_Attention(nn.Module):
	def __init__(self):
		super(Space_Attention, self).__init__()
		self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3)

	def forward(self, x):
		max_pool_x, _ = torch.max(x, dim=1, keepdim=True)  # 沿着通道维度进行最大池化
		avg_pool_x = torch.mean(x, dim=1, keepdim=True)  # 沿着通道维度进行平均池化
		# 沿着通道维度进行拼接
		pool_x = torch.cat([avg_pool_x, max_pool_x], dim=1)
		# 卷积
		pool_x = self.conv(pool_x)
		return F.sigmoid(pool_x)


class CBAM_ResNet_Element(nn.Module):
	expansion = 4

	def __init__(self, in_channel, out_channel, stride=1):
		super(CBAM_ResNet_Element, self).__init__()
		# 基础残差网络Res-50
		self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channel)

		self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channel)

		self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion * out_channel)

		# 通道注意力
		self.channel_attention = Channel_Attention(self.expansion * out_channel)
		# 空间注意力
		self.space_attention = Space_Attention()
		self.shortcut = nn.Sequential()
		if stride != 1 or in_channel != self.expansion * out_channel:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channel, self.expansion * out_channel, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * out_channel)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		channel_score = self.channel_attention(out)
		out = out * channel_score
		space_score = self.space_attention(out)
		out = out * space_score
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class CBAM_ResNets(nn.Module):
	def __init__(self, image_channel=1):
		super(CBAM_ResNets, self).__init__()
		self.block_list = [4, 3, 6, 3]
		self.start_channel = 64

		self.conv = nn.Conv2d(image_channel, self.start_channel, kernel_size=7, stride=2, padding=4, bias=False)
		self.maxPool = nn.MaxPool2d(3, stride=2)

		self.layer1 = self._new_layer(64, self.block_list[0], stride=1)
		self.layer2 = self._new_layer(128, self.block_list[1], stride=2)
		self.layer3 = self._new_layer(256, self.block_list[2], stride=2)
		self.layer4 = self._new_layer(512, self.block_list[3], stride=2)
		self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Sequential(
			nn.Linear(2048, 2),
			nn.Softmax()
		)

	def _new_layer(self, out_channel, block_num, stride):
		strides = [stride] + [1] * (block_num - 1)
		layers = []
		for stride in strides:
			cbam = CBAM_ResNet_Element(self.start_channel, out_channel, stride)
			layers.append(cbam)
			self.start_channel = out_channel * cbam.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		x = F.relu(self.maxPool(self.conv(x)))
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.avgPool(x)
		x = torch.flatten(x, 1)
		out = self.fc(x)
		return out


def train_cbam_resnet(epochs=10):
	if os.path.exists(cbam_model_path):
		cbam_model.load_state_dict(torch.load(cbam_model_path))
	cbam_model.train()
	for epoch in range(epochs):
		for i, (image, label) in enumerate(train_dataloader):
			image = image.view(environment.batch_size, -1, 224, 224).float().to(environment.device)
			label = label.view(environment.batch_size, -1).float().to(environment.device)
			output = cbam_model(image)
			loss = bce_criterion(output, label)
			cbam_optimizer.zero_grad()
			loss.backward()
			cbam_optimizer.step()
			cbam_scheduler.step(loss)
			# 记录损失到 TensorBoard
			writer.add_scalar('Loss/Train', loss.item(), epoch)

			# 可选：记录模型权重的直方图
			for name, param in cbam_model.named_parameters():
				writer.add_scalar('Train Loss', loss.item(), epoch)
				writer.flush()
			if i == 0:
				print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
		torch.save(cbam_model.state_dict(), cbam_model_path)


def test_cbam_resnet():
	cbam_model.eval()
	all_preds = []  # 存储所有预测的类别
	all_labels = []  # 存储所有标签
	if os.path.exists(cbam_model_path):
		cbam_model.load_state_dict(torch.load(cbam_model_path))
	for i, (image, label) in enumerate(test_dataloader):
		image = image.view(environment.batch_size, -1, 224, 224).float().to(environment.device)
		label = label.view(environment.batch_size, -1).float().to(environment.device)
		output = cbam_model(image)
		preds = torch.argmax(output, dim=1)  # 预测标签
		label = torch.argmax(label, dim=1)  # 真实标签
		if environment.device == 'cuda':
			all_preds.extend(preds.cpu().numpy())
			all_labels.extend(label.cpu().numpy())
		else:
			all_preds.extend(preds.numpy())
			all_labels.extend(label.numpy())

	# 计算各类评价指标
	accuracy = accuracy_score(all_labels, all_preds)
	precision = precision_score(all_labels, all_preds, average='weighted')  # 'micro', 'macro', 'weighted'等
	recall = recall_score(all_labels, all_preds, average='weighted')
	f1 = f1_score(all_labels, all_preds, average='weighted')

	# 打印评价指标
	print(f'Accuracy: {accuracy:.4f}')
	print(f'Precision: {precision:.4f}')
	print(f'Recall: {recall:.4f}')
	print(f'F1 Score: {f1:.4f}')


if __name__ == '__main__':
	cbam_model_path = "./save/cbam.pth"
	environment = test_environment
	# environment = train_environment
	dataset = ImageDataSet()
	train_size = int(0.8 * len(dataset))
	test_size = len(dataset) - train_size
	train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
	train_dataloader = DataLoader(train_dataset, batch_size=environment.batch_size, shuffle=True, drop_last=True)
	test_dataloader = DataLoader(test_dataset, batch_size=environment.batch_size, shuffle=True)
	bce_criterion = BCELoss()
	cbam_model = CBAM_ResNets().to(environment.device)
	cbam_optimizer = Adam(cbam_model.parameters(), lr=environment.lr)  # optimizer
	cbam_scheduler = lr_scheduler.ReduceLROnPlateau(cbam_optimizer, mode='min', factor=0.1, patience=10)
	print("start ...")
	writer = SummaryWriter('logs/cbam')
	# train_cbam_resnet(5)
	writer.close()
	test_cbam_resnet()
