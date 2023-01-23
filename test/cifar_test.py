import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchviz import make_dot
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn import functional as F
from torchvision.io import read_image


PATH_TRAIN_LABEL_CSV = r"D:\Onur\Projects\dog_vs_cat\data\train_label\image_label.csv"
PATH_TRAIN = r"D:\Onur\Projects\dog_vs_cat\data\train"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# from torchxray import Xray
# from torchxray.inspect import TraceArchitecture

# X = next(iter(trainloader))[0]
# X = X.to(device)
# print(X.size())
# x = X[0].resize(1, *X[0].shape)
# print('küçük x:', x.shape, type(x))

# xray = Xray(model, input_tensor=x, xray_id='reference-1')
#
# xray.initialize()

def trainer(model, optimizer, loss_func, trainloader):
	start_time = time.time()

	loss_history = []
	accuracy_history = []

	for epoch in range(3):  # loop over the dataset multiple times

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
			report_interval = 10
			print(f'[{epoch + 1}, {i + 1}] -- loss: {loss.item():.3f} -- acc: {acc:.3f}')
			if i % report_interval == 0:  # print every 2000 mini-batches
				# print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / report_interval:.3f} -- acc: ')
				# xray.take_graph(x, batch_num=epoch * len(trainloader) + i, show_plot=False, save_plot=True)
				running_loss = 0.0


	print('Finished Training')

	end_time = time.time()
	print('Elapsed time:', int((end_time - start_time) / 60), 'minutes')

# todo: save image diyip custom image save edebilelim /main_output_folder/output_folder/image/custom_images/.


if __name__ == '__main__':
	device = 'cuda' if torch.cuda.is_available() is True else 'cpu'
	# device = 'cpu'
	print('Device:', device)

	import torchvision
	import torchvision.transforms as transforms

	batch_size = 16
	transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
											download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
											  shuffle=True, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat',
			   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# model = PlainCNN()
	model = Net()
	model = model.to(device)

	loss_func = nn.CrossEntropyLoss()
	print('loss', loss_func)
	optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
	print(optimizer)

	trainer(model=model, optimizer=optimizer, loss_func=loss_func, trainloader=trainloader)

