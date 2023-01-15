from torchxray import Xray
from torchxray.inspect import TraceArchitecture


if __name__ == '__main__':
	import torch
	from torch import nn


	class MyModel(nn.Module):
		def __init__(self):
			super(MyModel, self).__init__()

			self.relu = nn.ReLU()
			self.tanh = nn.Tanh()
			self.sigmoid = nn.Sigmoid()
			self.cl1 = nn.Conv2d(1, 3, 3)
			self.cl2 = nn.Conv2d(3, 6, 3)
			self.cl3 = nn.Conv2d(6, 10, 5)

			self.flatten = torch.flatten
			self.fc1 = nn.Linear(24 * 24 * 10, 100)
			self.fc2 = nn.Linear(100, 100)
			self.fc3 = nn.Linear(100, 1)

		def forward(self, x):
			x = self.relu(self.cl1(x))
			x = self.relu(self.cl2(x))
			x = self.tanh(self.cl3(x))

			x = self.flatten(x, 0)

			x = self.relu(self.fc1(x))
			x = self.tanh(self.fc2(x))
			x = self.tanh(self.fc2(x))
			x = self.sigmoid(self.fc3(x))

			return x


	model = MyModel()
	x = torch.randn((1, 32, 32))

	# tarc = TraceArchitecture(model=model, input_tensor=x)
	# dict_graph_of_arc = tarc.get_core_architecture_graph_by_forward()
	# layers_ordered = tarc.get_core_architecture_list_by_forward()

	xray = Xray(model=model, input_tensor=x)
	xray.initialize()

	for i in range(3):
		xray.take_graph(X=torch.randn((1, 32, 32)), batch_num=i*10, show_plot=True, save_plot=True)

	print()
	# for key in dict_graph_of_arc.keys():
	# 	print(key.get_repr(), '->', end=' ')
	# 	for sub in dict_graph_of_arc[key]:
	# 		print(sub.get_repr(), '|')
	# 	print()

	# xray.trace_arc.plot_model(r'./exmaple_plot.png')

