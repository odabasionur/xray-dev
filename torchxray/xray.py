from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torchxray.inspect import TraceArchitecture
from torchxray.selection import map_module_type_selector

class Xray:
	def __init__(
			self,
			model: nn.Module,
			input_tensor: torch.Tensor = None,
			input_size: tuple = None,
			output: torch.Tensor = None,
			layer_list_to_inspect: list[str] = None,
			take_graph_of_activations: bool = True,
			take_graph_of_grad: bool = False,
			verbose: int = 0,
	):
		"""
		:attr dict_layer_output_draft:
		{
			'conv2d-0': {'weight': True, 'output': False, 'grad': False},
			'relu-0': {'weight': False, 'output': True, 'grad': False},
			'conv2d-1': {'weight': True, 'output': False, 'grad': False},
			'relu-1': {'weight': False, 'output': True, 'grad': False},
			'flatten': {'weight': False, 'output': False, 'grad': False},
			'linear-0': {'weight': True, 'output': False, 'grad': False},
			'relu-2': {'weight': False, 'output': True, 'grad': False},
			'linear-1': {'weight': True, 'output': False, 'grad': False},
			'sigmoid-0': {'weight': False, 'output': True, 'grad': False},
		}
		"""

		self.model = model
		self.input = None
		self.input_size = None
		self.output = output
		self.validate_input_or_output(input_tensor=input_tensor, input_size=input_size, output=output)

		self.user_marked_layers_to_inspect = layer_list_to_inspect  # Defined by user
		self.layers_to_inspect = []  # ['conv1', 'conv2', 'fc1']		# Defined by user if possible else by TraceArchitecture

		# Objects
		self.trace_arc = TraceArchitecture(
			model=self.model,
			input_tensor=self.input,
			input_size=input_size,
			output=self.output
		)

		self.layer_draft_example = {
			'conv': {'weight': True, 'output': False, 'grad': False},
			'linear': {'weight': True, 'output': False, 'grad': False},
			'activation': {'weight': False, 'output': True, 'grad': False},
			'unknown': {'weight': False, 'output': False, 'grad': False},
		}
		self.dict_layer_output_draft: dict = {}
		self.layer_name_module_map = OrderedDict()
		self.dict_layer_output_plan: dict = {}
		# **Pay attention not miss get the model in train mode!**
		if not self.model.training:
			self.model.train()

	def validate_input_or_output(self, input_tensor, input_size, output):
		if isinstance(input_tensor, torch.Tensor):
			self.input = input_tensor
		else:
			if isinstance(input_size, tuple):
				input_tensor = torch.ones(*input_size)
				self.input = input_tensor
			elif isinstance(output, torch.Tensor):
				if output.grad_fn is None:
					raise ValueError(
						"One of these two parameters input_tensor or input must be other than None. "
						"Or output's grad_fn attribute should not be None")
			else:
				raise ValueError('One of these three parameters input_tensor, input or output must be other than None')

	def initialize(self):
		"""
		This is further step of __init__(). It extracts the network architecture and handle defining attributes which
		are required when taking graphs. Since some procedures can also be done by user (like extracting graph), it
		prevents recurrent processes
		"""
		self.prepare_layer_output_draft()
		self.convert_draft_to_obj()
		self.add_selector_to_plan()

	def take_graph(self, x: torch.Tensor):
		"""
		Main method which manages other functions etc. to
			* Get defined layers to inspect
				* Get desired result to save
			* Use required methods to obtain desired result (may use forward or
			backward hooks or directly model parameters)
			* Save the results properly to be used later
		:return:
		"""
		inter = Interpreter(self.model, layer_name_map=self.layer_name_module_map)
		inter.dict_layer_output_plan = self.dict_layer_output_plan.copy()
		inter.list_layers_ordered = self.trace_arc.get_core_architecture_list_by_forward()
		inter.take_graph(x)

	def get_architecture_graph(self):
		return self.trace_arc.get_core_architecture_graph_by_forward()

	def prepare_layer_output_draft(self):
		layers = self.trace_arc.get_core_architecture_list_by_forward()
		self.dict_layer_output_draft = {
			layer.name: self.layer_draft_example.get(layer.module_type, self.layer_draft_example.get('unknown', None))
			for layer in layers}
		self.layer_name_module_map = {
			layer.name: layer
			for layer in layers}

	def convert_draft_to_obj(self):
		self.dict_layer_output_plan = {
			self.layer_name_module_map.get(layer_name): output_draft
			for layer_name, output_draft in self.dict_layer_output_draft.items()
		}

	def add_selector_to_plan(self):
		dict_selector_added = self.dict_layer_output_plan.copy()
		for module in self.dict_layer_output_plan.keys():
			if module.module_type == 'activation':
				selector_cls = map_module_type_selector[module.parent.module_type]['random']
			else:
				selector_cls = map_module_type_selector[module.module_type]['random']

			print(module, module.output_size)
			dict_selector_added[module]['selector'] = selector_cls(tensor_size=module.output_size)

		self.dict_layer_output_plan = dict_selector_added

	def interpret_user_parameters(self):
		pass

	def set_layers_to_inspect(self):
		if len(self.user_marked_layers_to_inspect) > 0:
			# todo: list içeriği ayrıca kontrol edilebilir (user_marked_layers_to_inspect).
			self.layers_to_inspect = self.user_marked_layers_to_inspect

		else:
			self.trace_arc.get_core_architecture_from_grad_fn()


class Interpreter:
	def __init__(self, model: nn.Module, layer_name_map: OrderedDict):
		self.model = model
		self.layer_name_map = layer_name_map.copy()
		self.dict_layer_output_plan: OrderedDict = OrderedDict()
		self.layer_output_map: OrderedDict
		self.list_layers_ordered: list = []

		self.hooked_modules = np.array([])		# todo: bunun farklı bir kullanımını oluşturmak lazım
		self.dict_output = {}

	def __set_dict_layer_output(self, dict_layer_plan):
		if isinstance(dict_layer_plan, OrderedDict):
			self.dict_layer_output_plan = dict_layer_plan.copy()

	def __set_layer_output_map(self):
		layer_output_map = OrderedDict()
		for layer_name, output_dict in self.dict_layer_output_plan.items():
			layer_output_map[self.layer_name_map[layer_name]]

	def take_graph(self, x: torch.Tensor):
		layers_output = [layer for layer in self.list_layers_ordered if self.dict_layer_output_plan[layer]['output']]
		layers_weight = [layer for layer in self.list_layers_ordered if self.dict_layer_output_plan[layer]['weight']]
		layers_grad = [layer for layer in self.list_layers_ordered if self.dict_layer_output_plan[layer]['grad']]

		d_ = self.hook_forward(
			input_tensor=x,
			modules_to_hook_outputs=layers_output,
			modules_to_hook_weights=layers_weight)

		dict_pruned = OrderedDict()
		for module, plan in self.dict_layer_output_plan.items():
			print(module)
			if plan['weight']:
				tensor_pruned = plan['selector'].make_selection(d_[module]['weight'])
				dict_pruned[module]['weight'] = tensor_pruned

			if plan['output']:
				tensor_pruned = plan['selector'].make_selection(d_[module]['output'])
				dict_pruned[module]['output'] = tensor_pruned

		print('PRUNEDD')
		print(dict_pruned)

		return dict_pruned

	def hook_forward(self, input_tensor: torch.Tensor, modules_to_hook_outputs: list, modules_to_hook_weights: list):

		unique_modules_to_hook = []
		module_names_to_hook_output = []
		module_names_to_hook_weight = []

		for data_module in modules_to_hook_outputs:
			if data_module.name not in module_names_to_hook_output:
				module_names_to_hook_output.append(data_module.name)
			if data_module.module_obj not in unique_modules_to_hook:
				unique_modules_to_hook.append(data_module.module_obj)
		for data_module in modules_to_hook_weights:
			if data_module.name not in module_names_to_hook_weight:
				module_names_to_hook_weight.append(data_module.name)
			if data_module.module_obj not in unique_modules_to_hook:
				unique_modules_to_hook.append(data_module.module_obj)

		hook_handles = []
		dict_module_outputs = OrderedDict()
		self.hooked_modules = np.array([])

		with torch.no_grad():
			def add_forward_hook():
				def hook(module, input_, output):
					module_common_name = module.__class__.__name__
					module_count = len(self.hooked_modules[self.hooked_modules == module_common_name])
					self.hooked_modules = np.append(self.hooked_modules, module_common_name)
					module_name = f'{module_common_name}-{module_count}'
					dict_sub_module_output = {}
					if module_name in module_names_to_hook_output:
						dict_sub_module_output['output'] = output.detach()
					if module_name in module_names_to_hook_weight:
						dict_sub_module_output['weight'] = list(module.parameters())[0].data

					if len(dict_sub_module_output) > 0:
						dict_module_outputs[self.layer_name_map[module_name]] = dict_sub_module_output

				return hook

		self.model.eval()

		for module in unique_modules_to_hook:
			hook_handles.append(module.register_forward_hook(add_forward_hook()))
		self.model(input_tensor)

		for i, hook_handle in enumerate(hook_handles):
			hook_handle.remove()

		if not self.model.training:
			self.model.train()

		return dict_module_outputs


class Grapher:
	def __init__(self, data_module):
		self.data_module = data_module

	def take_graph(self):
		"""
		pass
		:return:
		"""
