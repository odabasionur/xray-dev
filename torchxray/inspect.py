from collections import OrderedDict
from typing import Optional
import warnings
import graphviz
import numpy as np
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


class DataModule:
	def __init__(self, module, input_size, output_size, name=None, parent=None):
		self.module_obj: nn.Module = module
		self.name = name
		self.module_type = None
		self.input_size = input_size
		self.output_size = output_size
		self.parent: Optional[nn.Module] = parent
		self.child: Optional[nn.Module] = None
		self.root = True if self.parent is None else False
		self.leaf = True if self.child is None else False

		self._set_name()
		self._set_module_type()

	def _set_name(self):
		if self.name is None:
			try:
				self.name = self.module_obj.__str__().split('(')[0]
			except:
				if isinstance(self.module_obj, nn.Conv2d):
					self.name = 'Conv2d'
				elif isinstance(self.module_obj, nn.Linear):
					self.name = 'Linear'
				else:
					self.name = 'No-name'

	def _set_module_type(self):
		"""
		**Strong assumption**:
		"""
		try:
			self.module_type = self.module_obj.__module__.split('.')[-1]
		except Exception as e:
			warnings.warn(f'module type of layer {self.name} & {self.module_obj} is unknown')
			self.module_type = 'unknown'

	def __repr__(self):
		return f'{self.name}'


class TraceArchitecture:
	def __init__(
			self,
			model: Optional[nn.Module] = None,
			input_tensor: Optional[torch.Tensor] = None,
			input_size: Optional[tuple] = None,
			output: Optional[torch.Tensor] = None):

		self.model = model
		self.input = input_tensor
		self.input_size = input_size
		if isinstance(input_tensor, torch.Tensor):
			self.input = input_tensor
		else:
			if isinstance(self.model, nn.Module) & isinstance(self.input_size, tuple):
				input_tensor = self.model(torch.ones(*self.input_size))
				self.input = input_tensor
		self.output = output

		self.dict_graph_module = OrderedDict()
		self.dict_graph_module_filtered = OrderedDict()
		self.dict_forward_draft = OrderedDict()
		self.dict_graph_arc = OrderedDict()
		self.arr_modules = np.array([])

	@staticmethod
	def _add_node_if_not_exist(dict_modify, key):
		if dict_modify.get(key) is None:
			dict_modify[key] = []

	@staticmethod
	def is_keyword_necessary(keyword):
		key = keyword.name() if keyword else keyword
		non_core_layer_variables = ['addbackward', 'unsqueezebackward', 'squeezebackward', 'accumulategrad']
		if isinstance(key, str):
			# If this if statement need to be change in the future, it must be
			# considered that the condition of `keyword == None` is possible
			for non_layer_var in non_core_layer_variables:
				if non_layer_var in key.lower():
					return False
		else:
			return False
		return True

	def travel_through_grad_fn(self, grad_fn, child_fn):
		"""
		Follows output's grad_fn through backpropagation recursively to extract the network architecture
		"""
		if child_fn is None:
			# The last/leaf node. So this is the child_fn
			self._add_node_if_not_exist(self.dict_graph_module, grad_fn)
			self.travel_through_grad_fn(grad_fn.next_functions, grad_fn)

		if isinstance(grad_fn, tuple):
			if len(grad_fn) == 0:
				return None

			for i, func_comp in enumerate(grad_fn):
				current_func = func_comp[0]
				if current_func is None:
					continue

				self._add_node_if_not_exist(self.dict_graph_module, current_func)
				self.dict_graph_module[child_fn].append(current_func)
				self.travel_through_grad_fn(current_func.next_functions, child_fn)

	def filter_graph_dict(self, key, child_key):
		"""
		Removes unnecessary functions (like squeeze, addition etc. )from architecture dictionary recursively created by grad_fn's
		"""
		if child_key is None:
			# if leaf node
			self._add_node_if_not_exist(self.dict_graph_module_filtered, key)
			for parrent_key in self.dict_graph_module[key]:
				self.filter_graph_dict(parrent_key, key)

		elif key is None:
			return None

		elif self.is_keyword_necessary(key):
			self._add_node_if_not_exist(self.dict_graph_module_filtered, child_key)
			self.dict_graph_module_filtered[child_key].append(key)
			for parrent_key in self.dict_graph_module[key]:
				self.filter_graph_dict(parrent_key, key)

		else:
			for parrent_key in self.dict_graph_module[key]:
				self.filter_graph_dict(parrent_key, child_key)

	def get_architecture_in_detail_from_grad_fn(self) -> dict:
		"""
		Extract architecture by output's grad_fn and returns it
		"""
		self.travel_through_grad_fn(grad_fn=self.output.grad_fn, child_fn=None)
		return self.dict_graph_module

	def get_core_architecture_from_grad_fn(self) -> dict:
		"""
		Returns only necessary functions extracted by `get_architecture_in_detail_from_grad_fn()`
		"""
		if len(self.dict_graph_module) == 0:
			self.get_architecture_in_detail_from_grad_fn()

		self.filter_graph_dict(key=self.output.grad_fn, child_key=None)
		return self.dict_graph_module_filtered

	def hook_forwards(self):
		hooks = []
		self.dict_forward_draft = OrderedDict()
		self.arr_modules = np.array([])

		with torch.no_grad():
			def add_forward_hook(module):
				"""
				For usage of extracting architecture
				"""
				def hook(module, input_, output):
					dict_module_info = OrderedDict()
					module_common_name = module.__class__.__name__
					module_count = len(self.arr_modules[self.arr_modules == module_common_name])
					dict_module_info['module'] = module
					dict_module_info['input_size'] = input_[0].size()
					dict_module_info['output_size'] = output[0].size()
					self.dict_forward_draft[f'{module_common_name}-{module_count}'] = dict_module_info.copy()
					self.arr_modules = np.append(self.arr_modules, module_common_name)

				hooks.append(module.register_forward_hook(hook))

			self.model.eval()
			self.model.apply(add_forward_hook)
			self.model(self.input)

		for i, hook in enumerate(hooks):
			hook.remove()

		if not self.model.training:
			self.model.train()

	def extract_info(self):
		"""module, input_size, output_size"""
		is_prev_layer = True
		prev_module = None
		for module_order, key in enumerate(list(self.dict_forward_draft.keys())[:-1]):
			dict_of_module = self.dict_forward_draft[key]
			module = dict_of_module.get('module')
			module_name = key
			input_size = dict_of_module.get('input_size')
			output_size = dict_of_module.get('output_size')
			current_module = DataModule(
				module=module, input_size=input_size, output_size=output_size, name=module_name, parent=prev_module)
			if prev_module:
				self._add_node_if_not_exist(self.dict_graph_arc, prev_module)
				self.dict_graph_arc[prev_module].append(current_module)
			prev_module = current_module

	def extract_core_architecture_by_forward(self):
		self.dict_graph_arc = OrderedDict()
		self.hook_forwards()
		self.extract_info()

	def get_core_architecture_graph_by_forward(self, from_scratch=False) -> OrderedDict:
		"""
		Returns the dictionary of layers in the order pytorch process
		from_scratch: bool
			True to re-extract architecture
		"""

		if from_scratch:
			self.extract_core_architecture_by_forward()
			return self.dict_graph_arc.copy()

		if isinstance(self.dict_graph_arc, dict):
			if len(self.dict_graph_arc) > 0:
				return self.dict_graph_arc.copy()

		self.extract_core_architecture_by_forward()
		return self.dict_graph_arc.copy()

	def get_core_architecture_list_by_forward(self, from_scratch=False) -> list:
		"""
		Returns the list of layers in the order pytorch process
		from_scratch: bool
			True to re-extract architecture
		"""
		dict_graph_arc = self.get_core_architecture_graph_by_forward(from_scratch=from_scratch)
		dict_layers_ordered = OrderedDict()

		for layer, child_layers in dict_graph_arc.items():
			dict_layers_ordered[layer] = None
			for child_layer in child_layers:
				dict_layers_ordered[child_layer] = None

		return list(dict_layers_ordered.keys())

	def plot_model(self, path: Optional[str] = None):
		dict_graph_arc = self.get_core_architecture_graph_by_forward()
		diagraph = graphviz.Digraph(comment='plot_model')
		diagraph.graph_attr.update({'rankdir': 'TB'})

		for node, sub_nodes in dict_graph_arc.items():
			diagraph.node(node.__repr__(), label=node.__repr__())
			for sub_node in sub_nodes:
				print(node.__repr__(), sub_node.__repr__())
				diagraph.edge(node.__repr__(), sub_node.__repr__())
				# diagraph.edge(node.get_repr(), sub_node.get_repr(), color='red', fontcolor='red', fontsize="8")

		diagraph.view(path)


class Hooker:
	def __init__(self):
		self.dict_of_output = OrderedDict()
		self.hooks: list[RemovableHandle] = []

	def forward_hook_up_draft(self, dict_of_layer: Optional[OrderedDict] = None):
		"""
		For usage of extracting architecture
		"""
		def hook(module, input_, output):
			d_output = dict_of_layer if isinstance(dict_of_layer, OrderedDict) else self.dict_of_output
			d_output['module'] = module
			d_output['input_size'] = input_.size()
			d_output['output_size'] = output.size()
		return hook

	def forward_hook_up_layer(self, dict_of_layer: Optional[OrderedDict] = None):
		"""
		For usage of inspecting module, input, output of hooked modules as forwarding
		"""
		def hook(module, input_, output):
			d_output = dict_of_layer if isinstance(dict_of_layer, OrderedDict) else self.dict_of_output
			d_output['module'] = module
		return hook

	def backward_hook_up_layer(self, dict_of_layer: Optional[OrderedDict] = None):
		"""
		For usage of inspecting module, input, output of hooked modules as forwarding
		"""
		def hook(module, input_, output):
			d_output = dict_of_layer if isinstance(dict_of_layer, OrderedDict) else self.dict_of_output
			d_output['module'] = module
		return hook

	def _remove_hooks(self):
		for hook in self.hooks:
			hook.remove()


class DataGradFn:
	def __init__(self, fn, parent=None):
		self.fn = fn
		self.leaf = False
		self.root = False
		self.children = None
		self.parent = parent

	@property
	def name(self):
		return self.fn.name()

	@property
	def size(self):
		return self.fn.size()

	@property
	def label(self):
		return

	def __str__(self):
		return self.fn.__str__()

	def __repr__(self):
		return f'{self.__str__()}'
