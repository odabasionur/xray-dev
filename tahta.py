from collections import OrderedDict

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
		self.fc1 = nn.Linear(28*28*6, 1)

	def forward(self, x):
		x = self.relu(self.cl1(x))
		x = self.relu(self.cl2(x))
		print(x.size())

		x = torch.flatten(x, 0)
		print(x.size())

		x = F.softmax(self.fc1(x))
		return x


activation = {}
def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook


model = MyModel()
# model.fc2.register_forward_hook(get_activation('fc2'))
x = torch.randn((1, 32, 32))
# output = model(x)
# print(output)


output = model(x)
output.grad


def travel_through_grad_fn(self, grad_fn, child_fn):
	"""
	Since torch.Variable.grad_fn.next_functions keep functions family
	"""
	if child_fn is None:
		print('child None')
		# The last/leaf node. So this is the child_fn
		dgf_grad_fn = DataGradFn(grad_fn)
		self._add_node_if_not_exist(self.dict_graph_module, dgf_grad_fn)
		print('dict graph module')
		print(self.dict_graph_module)
		self.travel_through_grad_fn(grad_fn.next_functions, dgf_grad_fn)

	if isinstance(grad_fn, tuple):
		if len(grad_fn) == 0:
			#             dict_graph[child_fn.__str__()].append(None)
			return None

		for i, func_comp in enumerate(grad_fn):
			current_func = func_comp[0]
			if current_func is None:
				continue

			dgf_current = DataGradFn(current_func)
			# dgf_child = DataGradFn(child_fn)

			self._add_node_if_not_exist(self.dict_graph_module, dgf_current)
			print(self.dict_graph_module)
			print(dgf_current)
			self.dict_graph_module[child_fn].append(dgf_current)
			# dgf_child.parent = dgf_current
			self.travel_through_grad_fn(current_func.next_functions, DataGradFn(child_fn, dgf_current))
