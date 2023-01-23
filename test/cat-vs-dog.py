import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torchviz import make_dot
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn import functional as F
from torchvision.io import read_image


PATH_TRAIN_LABEL_CSV = r"D:\Onur\Projects\dog_vs_cat\data\train_label\image_label.csv"
PATH_TRAIN = r"D:\Onur\Projects\dog_vs_cat\data\train"


device = 'cuda' if torch.cuda.is_available() is True else 'cpu'
# device = 'cpu'
print('Device:', device)


class DCDataset(Dataset):

	def __init__(self, annotation_file: str, img_dir: str, transform=None, target_transform=None, with_label=True):
		self.df_label = pd.read_csv(annotation_file)
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.df_label)

	def __getitem__(self, idx):
		#         df_img_row = self.df_label.iloc[idx, :]
		#         img_path = os.path.join(self.img_dir, df_img_row[0])

		#         image = read_image(img_path)
		#         label = df_img_row[1]
		img_path = os.path.join(self.img_dir, self.df_label.iloc[idx, 0])

		image = read_image(img_path).float()
		label = self.df_label.iloc[idx, 1]

		if self.transform:
			image = self.transform(image)

		return image, label


class PlainCNN(nn.Module):
	def __init__(self):
		super(PlainCNN, self).__init__()

		self.relu = nn.ReLU()

		# Static Layers
		self.maxpool_std2 = nn.MaxPool2d(2, 2)
		# self.flatten = nn.Flatten(-1)

		# Conv Layers
		self.conv1 = nn.Conv2d(3, 16, 5)
		self.conv2 = nn.Conv2d(16, 32, 5, padding=1)  # 106 ( / 2 = 58)
		self.conv3 = nn.Conv2d(32, 64, 5)  # 52  ( / 2 = 26)
		# self.conv4 = nn.Conv2d(128, 196, 3)  # 52  ( / 2 = 26)

		# Fully-Connected Layers
		self.linear1 = nn.Linear(25 * 25 * 64, 500)  # 26*26*24
		self.linear2 = nn.Linear(500, 100)
		self.linear3 = nn.Linear(100, 2)

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.maxpool_std2(x)

		x = self.relu(self.conv2(x))
		x = self.maxpool_std2(x)

		x = self.relu(self.conv3(x))
		x = self.maxpool_std2(x)

		x = nn.Flatten()(x)
		x = self.relu(self.linear1(x))

		x = self.relu(self.linear2(x))

		x = self.linear3(x)

		return x


class ReferenceNetwork(nn.Module):
	def __init__(self):
		super(ReferenceNetwork, self).__init__()

		# Static Layers
		self.maxpool_std2 = nn.MaxPool2d(2, 2)
		self.flatten = nn.Flatten()
		self.relu = nn.ReLU()
		self.dropout10 = nn.Dropout(p=0.1)
		self.dropout15 = nn.Dropout(p=0.15)
		self.dropout20 = nn.Dropout(p=0.2)

		# Conv Layers
		self.conv1 = nn.Conv2d(3, 8, 5)
		self.conv2 = nn.Conv2d(8, 16, 7)  # 106 ( / 2 = 58)
		self.conv3 = nn.Conv2d(16, 24, 5)  # 52  ( / 2 = 26)
		self.conv4 = nn.Conv2d(32, 64, 3)  # 52  ( / 2 = 26)

		# Fully-Connected Layers
		self.linear1 = nn.Linear(24 * 24 * 24, 4096)  # 26*26*24
		# self.linear1 = nn.Linear(4096, 1000)  # 26*26*24
		self.linear2 = nn.Linear(4096, 512)
		self.linear3 = nn.Linear(512, 2)

	def forward(self, x):
		x = self.relu(self.conv1(x))
		x = self.maxpool_std2(x)

		x = self.relu(self.conv2(x))
		x = self.maxpool_std2(x)

		x = self.relu(self.conv3(x))
		x = self.maxpool_std2(x)

		# x = self.relu(self.conv4(x))
		# x = self.maxpool_std2(x)

		x = self.flatten(x)
		x = self.relu(self.linear1(x))
		# x = self.dropout15(x)
		#
		x = self.relu(self.linear2(x))
		# x = self.dropout10(x)
		#
		x = self.linear3(x)

		return x


def trainer(model, optimizer, loss_func, trainloader):
	from torchxray import Xray
	from torchxray.inspect import TraceArchitecture

	image_cat = read_image(r'D:\Onur\Projects\dog_vs_cat\data\train\cat.194.jpg').float()
	image_dog = read_image(r'D:\Onur\Projects\dog_vs_cat\data\train\dog.4249.jpg').float() 	# 6450, 8115,
	x = image_dog.resize(1, 3, 224, 224).to(device)
	print('x size', x.size())

	xray = Xray(model, input_tensor=x, xray_id='reference-9-dog3-nodropout')
	xray.initialize()
	print(xray.dict_layer_output_draft)

	xray.display.display_filter(tensor=x[0]/255, title='reference_image', show_plot=True, save_plot=True,
								path=os.path.join(xray.fm.output_image_dir, 'reference_image.png'))

	start_time = time.time()

	loss_history = []
	accuracy_history = []

	for epoch in range(10):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			model.train()
			# print('Epoch:', epoch, 'Batch:', i)
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(inputs)
			loss = loss_func(outputs, labels)

			output_labels = torch.argmax(outputs, 1)
			acc = ((output_labels == labels).sum() / len(labels)).item()
			accuracy_history.append(acc)

			loss.backward()
			optimizer.step()

			# print statistics
			loss_history.append(loss.item())
			running_loss += loss.item()
			report_interval = 50
			print(f'[{epoch + 1}, {i + 1}] -- loss: {loss.item():.3f} -- acc: {acc:.3f}')
			if i % report_interval == 0:  # print every 2000 mini-batches
				# print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / report_interval:.3f} -- acc: ')
				xray.take_graph(x, batch_num=epoch * len(trainloader) + i, show_plot=False, save_plot=True)
				running_loss = 0.0

			if i % 700 == 0:  # print every 2000 mini-batches

				plt.cla()
				plt.clf()
				sns.lineplot(y=accuracy_history, x=list(range(len(accuracy_history)))).set_title(f'Accuracy [{epoch + 1}, {i + 1}]')
				plt.show()
				sns.lineplot(y=loss_history, x=list(range(len(loss_history)))).set_title(f'Loss [{epoch + 1}, {i + 1}]')
				plt.ylim(-0.01, 1)
				plt.show()

	print('Finished Training')

	end_time = time.time()

	xray.display.display_filter(tensor=x[0]/255, title='reference_image2', show_plot=True, save_plot=True,
								path=os.path.join(xray.fm.output_image_dir, 'reference_image.png'))
	xray.fm.save_history(loss_history=loss_history, accuracy_history=accuracy_history)
	print('Elapsed time:', int((end_time - start_time) / 60), 'minutes')

# todo: save image diyip custom image save edebilelim /main_output_folder/output_folder/image/custom_images/.


if __name__ == '__main__':
	all_train_dataset = DCDataset(
		annotation_file=PATH_TRAIN_LABEL_CSV,
		img_dir=PATH_TRAIN)

	train_batch_size = 32
	trainloader = DataLoader(all_train_dataset, batch_size=train_batch_size, shuffle=True)

	# model = PlainCNN()
	model = ReferenceNetwork()
	model = model.to(device)

	loss_func = nn.CrossEntropyLoss()
	print('loss', loss_func)
	optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
	# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
	print(optimizer)

	trainer(model=model, optimizer=optimizer, loss_func=loss_func, trainloader=trainloader)

