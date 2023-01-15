from abc import ABC
import warnings
from typing import Union
import torch
import numpy as np


class BaseDimensionSelector(ABC):
	def __init__(self, tensor_size: Union[torch.Size, tuple[int]], parent_module_type: str = None, random_seed=42):
		"""
		tensor_size: Union[torch.Size, tuple[int]]
			The expected size of tensor exposed to make selection
		module_type: str
			One of {'conv2d', 'linear', 'activation'}
		parent_module_type: str
			This is specifically necessary for activation functions. Activation probably need to be interpreted as their
			parent modules
		random_seed: int
		"""
		self.tensor_size = tensor_size

	def check_dimension_number(self, tensor_size: iter):
		return

	def make_selection(self, tensor: torch.Tensor):
		return


class BaseFilterSelector(BaseDimensionSelector):
	"""
	For convolutional type modules
	"""
	def __init__(self, tensor_size: Union[torch.Size, tuple[int]], parent_module_type: str = None, random_seed=42):
		super(BaseFilterSelector, self).__init__(
			tensor_size=tensor_size,
			parent_module_type=parent_module_type,
			random_seed=random_seed)

		self.required_dim_num = 2
		self.extra_dim_num = len(self.tensor_size) - self.required_dim_num

	def check_dimension_number(self, tensor_size: iter):
		return

	def make_selection(self, tensor: torch.Tensor):
		return


class BaseNodeSelector(BaseDimensionSelector):
	"""
	For linear type modules
	"""
	def __init__(self, tensor_size: Union[torch.Size, tuple[int]], parent_module_type: str = None, random_seed=42):
		super(BaseNodeSelector, self).__init__(
			tensor_size=tensor_size,
			parent_module_type=parent_module_type,
			random_seed=random_seed)

		self.required_dim_num = 1
		self.max_node = 100*100
		self.extra_dim_num = len(self.tensor_size) - self.required_dim_num

	def check_dimension_number(self, tensor_size: iter):
		return

	def make_selection(self, tensor: torch.Tensor):
		return


class RandomFilterSelector(BaseFilterSelector):
	def __init__(self, tensor_size: Union[torch.Size, tuple[int]], parent_module_type: str = None, random_seed=42):
		super(RandomFilterSelector, self).__init__(
			tensor_size=tensor_size,
			parent_module_type=parent_module_type,
			random_seed=random_seed)

		# self.extra_dims = tensor_size[:self.extra_dim_num]
		self.extra_dims = [tensor_size[dim] for dim in range(self.extra_dim_num)]
		self.random_dimensions = [np.random.randint(0, dim) for dim in self.extra_dims]

	def make_selection(self, tensor: torch.Tensor):
		tensor_size = tensor.size()
		extra_dim_num = len(tensor_size) - self.required_dim_num

		if extra_dim_num == 0:
			if tensor_size[-1] == tensor_size[-2]:
				return tensor
			else:
				warnings.warn(f'Tensor is not proper for conv layer with dimensions of {tensor_size}')
				return None

		elif extra_dim_num < 0:
			warnings.warn(f'Tensor is not proper for conv layer with dimensions of {tensor_size}')
			return None

		elif extra_dim_num > 0:
			if tensor_size[-1] != tensor_size[-2]:
				warnings.warn(f'Tensor is not proper for conv layer with dimensions of {tensor_size}')
				return None

			for extra_dim in range(extra_dim_num):
				tensor = tensor[self.random_dimensions[extra_dim], :]
			return tensor


class RandomNodeSelector(BaseNodeSelector):
	def __init__(self, tensor_size: Union[torch.Size, tuple[int]], parent_module_type: str = None, max_node=64*64, random_seed=42):
		super(RandomNodeSelector, self).__init__(
			tensor_size=tensor_size,
			parent_module_type=parent_module_type,
			random_seed=random_seed)

		# self.extra_dims = tensor_size[:self.extra_dim_num]
		self.extra_dims = [tensor_size[dim] for dim in range(self.extra_dim_num)]
		self.random_dimensions = [np.random.randint(0, dim) for dim in self.extra_dims]

	def make_selection(self, tensor: torch.Tensor):
		tensor_size = tensor.size()
		extra_dim_num = len(tensor_size) - self.required_dim_num

		if extra_dim_num == 0:
			return tensor

		elif extra_dim_num < 0:
			warnings.warn(f'Tensor is not proper for conv layer with dimensions of {tensor_size}')
			return None

		elif extra_dim_num > 0:
			for extra_dim in range(extra_dim_num):
				tensor = tensor[self.random_dimensions[extra_dim], :]
				tensor = tensor[:self.max_node]
			return tensor


map_module_type_selector = {
	'conv': {
		'random': RandomFilterSelector
	},
	'linear': {
		'random': RandomNodeSelector
	}
}


if __name__ == '__main__':
	print(type(RandomNodeSelector))
	tensor_size = torch.Size([3, 1, 4, 5])
	x = torch.ones(tensor_size)
	rfs = RandomFilterSelector(tensor_size)
	new = rfs.make_selection(x)
	print(new.size())
