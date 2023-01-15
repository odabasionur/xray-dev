import torch
from PIL import Image
import matplotlib.pyplot as plt


class Display:

	@staticmethod
	def display_filter(tensor: torch.Tensor, title: str = ''):
		size_length = len(tensor.size())

		if size_length > 4:
			raise ValueError(f'Tensor given to display can not have more than 4 dimensions but given {tensor.size()}')

		elif size_length == 4:
			pass

		elif size_length == 3:
			plt.imshow(tensor.permute(1, 2, 0))
			plt.title(title)
			plt.show()

		elif size_length == 2:
			plt.imshow(tensor)
			plt.title(title)
			plt.show()

	@staticmethod
	def display_nodes(tensor: torch.Tensor, title: str = ''):

		if len(tensor.size()) == 1:
			node_num = tensor.size()[-1]
			edge_node_num = int(node_num ** (1/2))
			tensor = tensor[:edge_node_num ** 2]
			tensor = tensor.resize(edge_node_num, edge_node_num)
			plt.imshow(tensor)
			plt.title(title)
			plt.show()

		else:
			raise ValueError(
				f'Tensor given to display does not have proper dimension. It must consist of 1-dimension '
				f'but given, {tensor.size()}')
