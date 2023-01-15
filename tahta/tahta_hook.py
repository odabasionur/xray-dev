import torch
from torch import nn
from torch.nn import functional as F


class MyModel(nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()

		self.relu = nn.ReLU()
		self.cl1 = nn.Conv2d(1, 3, 3)
		self.cl2 = nn.Conv2d(3, 6, 3)
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(28 * 28 * 6, 1)

	def forward(self, x):
		x = self.relu(self.cl1(x))
		x = self.relu(self.cl2(x))

		x = torch.flatten(x, 0)

		x = F.softmax(self.fc1(x))
		return x


model = MyModel()
x = torch.randn((1, 32, 32))
output = model(x)



from collections import OrderedDict

dict_storage = OrderedDict()


def forward_hook_up_layer(dict_of_layer: dict):
	def hook(module, input_, output):
		dict_of_layer['module'] = module
		dict_of_layer['output'] = output

	return hook


def backward_hook_up_layer(dict_of_layer: dict):
	def hook(module, input_, output):
		dict_of_layer['module'] = module
		dict_of_layer['output'] = output

	return hook


h1 = model.register_forward_hook(forward_hook_up_layer(dict_storage))


