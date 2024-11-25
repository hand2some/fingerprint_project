"""
About the loading process of image data
the step including:
(1) read image
(2) get window image
(3) transform image
(4) get the label of image, default is [1,0]
(5) let data to dataset
(6) let dataset to dataloader
"""

import numpy
from torchvision import transforms
from setting import np, pd, os,torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# step 1
def read_img(base_image_dir: str, index: int) -> numpy.array:
	"""
	read image
	:param index: the index of image
	:return: the gray numpy array of image
	"""
	img_path = os.path.join(base_image_dir, f'{index}.bmp')
	img = Image.open(img_path).convert("L")  # 转为灰度图像
	# img.show()
	gray_array = np.array(img)
	return gray_array


# step 2
def get_window_img(gray_array: numpy.array, window_size=(224, 224), step=50):
	"""
	get the window image
	:param gray_array: the gray numpy array of image
	:param window_size: the size of window,include the width and height of window,the default is (224,224)
	:param step: the step of window,the default value is 50
	:return:
	"""
	gray_array_size = gray_array.shape  # (224,672)
	if gray_array_size[0] == window_size[0]:
		for i in range(step, gray_array_size[1] - window_size[1], step):
			yield gray_array[:, i:i + window_size[1]]


# step 3:the image data transform process
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.5], std=[0.5])
])


# step 4:define the image dataset
class ImageDataSet(Dataset):
	base_image_dir = '../data/A02-right-1-female-1'  # the basic directory path of images

	def __init__(self, nums=10, transform=None):
		"""
		the init function
		:param nums: the number of loading image
		:param transform: the transform process of image
		"""
		super(ImageDataSet, self).__init__()
		self.transform = transform
		self.data = []
		for i in range(1, nums + 1):
			gray_array = read_img(self.base_image_dir, i)
			for window_img in get_window_img(gray_array):
				self.data.append(window_img)
		self.data = np.array(self.data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img_data = self.data[index]
		label = torch.tensor([1, 0])  # the default label of image is [1,0]
		if self.transform:
			img_data = self.transform(img_data)
		return img_data, label


if __name__ == '__main__':
	dataset = ImageDataSet()
	for data in dataset:
		print(data[0].shape)
		print(data[1])
		break
# gray_array = read_img(1)
# for window_img in get_window_img(gray_array):
# 	print(window_img.shape)
