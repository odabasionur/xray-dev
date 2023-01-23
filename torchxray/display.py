import os
import torch
import matplotlib.pyplot as plt
import imageio


class Display:

	@staticmethod
	def display_filter(
			tensor: torch.Tensor,
			title: str = '',
			show_plot: bool = True,
			save_plot: bool = False,
			path: str = None,
			cmap: str = 'inferno'):


		size_length = len(tensor.size())

		if size_length > 4:
			raise ValueError(f'Tensor given to display can not have more than 4 dimensions but given {tensor.size()}')

		elif size_length == 4:
			pass

		elif size_length == 3:
			# plt.cla()
			# plt.clf()
			plt.imshow(tensor.permute(1, 2, 0).to('cpu'), cmap=cmap)
			plt.title(title)
			if save_plot:
				plt.savefig(path)
			if show_plot:
				plt.show()
			# todo: plt cache temizlemeyi deco yap

		elif size_length == 2:
			# plt.cla()
			# plt.clf()

			plt.imshow(tensor.to('cpu'), cmap=cmap)
			plt.title(title)
			if save_plot:
				plt.savefig(path)
			if show_plot:
				plt.show()

	@staticmethod
	def display_nodes(
			tensor: torch.Tensor,
			title: str = '',
			show_plot: bool = True,
			save_plot: bool = False,
			path: str = None,
			cmap: str = 'inferno'):

		# plt.cla()
		# plt.clf()

		if show_plot | save_plot:
			if len(tensor.size()) == 1:
				node_num = tensor.size()[-1]
				edge_node_num = int(node_num ** (1/2))
				tensor = tensor[:edge_node_num ** 2]
				tensor = tensor.resize(edge_node_num, edge_node_num)

			plt.imshow(tensor.to('cpu'), cmap=cmap)
			plt.title(title)
			if show_plot:
				plt.show()
			if save_plot:
				plt.savefig(path)

		else:
			raise ValueError(
				f'Tensor given to display does not have proper dimension. It must consist of 1-dimension '
				f'but given, {tensor.size()}')

	def create_gif(self, path, gif_filename, image_extension='png'):
		images = os.listdir(path)
		images = list(filter(lambda x: x.split('.')[-1].lower() == image_extension.lower(), images))
		with imageio.get_writer(os.path.join(path, gif_filename + '.gif'), mode='I') as writer:
			for image_name in images:
				image = imageio.imread(os.path.join(path, image_name))
				writer.append_data(image)

