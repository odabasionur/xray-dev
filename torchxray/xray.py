import uuid
from collections import OrderedDict
from typing import Union
import numpy as np
import torch
from torch import nn
from torchxray.inspect import TraceArchitecture
from torchxray.selection import map_module_type_selector
from torchxray.file_manager import FileManager
from torchxray.display import Display


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
			dont_create_folder: bool = False,
			output_directory: str = r'./xray_outputs/',
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
		self.dont_create_folder = dont_create_folder
		self.xray_id = uuid.uuid4().__str__().split('-')[-1]

		self.user_marked_layers_to_inspect = layer_list_to_inspect  # Defined by user
		self.layers_to_inspect = []  # ['conv1', 'conv2', 'fc1']		# Defined by user if possible else by TraceArchitecture

		# Objects
		self.trace_arc = TraceArchitecture(
			model=self.model,
			input_tensor=self.input,
			input_size=input_size,
			output=self.output
		)
		self.fm = FileManager(main_output_directory=output_directory, xray_id=self.xray_id)
		self.display = Display()

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
		print('initializinggg')
		self.prepare_layer_output_draft()
		self.convert_draft_to_obj()
		self.add_selector_to_plan()
		print('aloooooo')
		self.fm.create_sub_folders([data_module.name for data_module in self.dict_module_output_selector.keys()])

	def take_graph_(self, x: torch.Tensor, batch_num: int = 0):
		"""
		Main method which manages other functions etc. to
			* Get defined layers to inspect
				* Get desired result to save
			* Use required methods to obtain desired result (may use forward or
			backward hooks or directly model parameters)
			* Save the results properly to be used later
		:return:
		"""
		inter = Interpreter(self.model, layer_name_map=self.layer_name_module_map, xray_id=self.xray_id, file_manager=self.fm)
		inter.dict_layer_output_plan = self.dict_layer_output_plan.copy()
		inter.list_layers_ordered = self.trace_arc.get_core_architecture_list_by_forward()
		inter.take_graph(x, batch_num=batch_num)

	def take_graph(self, X: torch.Tensor, batch_num: int = 0, show_plot=True):

		modules_to_hook_outputs = [
			layer for layer in self.trace_arc.get_core_architecture_list_by_forward()
			if self.dict_layer_output_plan[layer]['output']]
		modules_to_hook_weights = [
			layer for layer in self.trace_arc.get_core_architecture_list_by_forward()
			if self.dict_layer_output_plan[layer]['weight']]
		# layers_grad = [layer for layer in self.list_layers_ordered if self.dict_layer_output_plan[layer]['grad']]

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
					data_module_ = self.layer_name_module_map[module_name]

					if module_name in module_names_to_hook_output:
						tensor_pruned = self.dict_module_output_selector[data_module_]['output'].\
							make_selection(output.detach())
						self.fm.save_data(
							tensor=tensor_pruned, xray_id=self.xray_id, layer_name=data_module_.name,
							output_type='output', batch_num=batch_num, extension='.pt')
						if show_plot:
							if data_module_.module_type == 'activation':
								module_type = data_module_.parent.module_type
							else:
								module_type = data_module_.module_type
							if module_type == 'conv':
								self.display.display_filter(
									tensor=tensor_pruned,
									title=f'{module_name} - {"x".join([str(s) for s in tensor_pruned.size()])}')
							elif module_type == 'linear':
								self.display.display_nodes(
									tensor=tensor_pruned,
									title=f'{module_name} - {"x".join([str(s) for s in tensor_pruned.size()])}')

					if module_name in module_names_to_hook_weight:
						tensor_pruned = self.dict_module_output_selector[data_module_]['weight']. \
							make_selection(list(module.parameters())[0].data)
						self.fm.save_data(
							tensor=tensor_pruned, xray_id=self.xray_id, layer_name=data_module_.name,
							output_type='weight', batch_num=batch_num, extension='.pt')
						if show_plot:
							if data_module_.module_type == 'activation':
								module_type = data_module_.parent.module_type
							else:
								module_type = data_module_.module_type
							if module_type == 'conv':
								self.display.display_filter(
									tensor=tensor_pruned,
									title=f'{module_name} - {"x".join([str(s) for s in tensor_pruned.size()])}')
							elif module_type == 'linear':
								self.display.display_nodes(
									tensor=tensor_pruned,
									title=f'{module_name} - {"x".join([str(s) for s in tensor_pruned.size()])}')
				return hook

		self.model.eval()
		for module in unique_modules_to_hook:
			hook_handles.append(module.register_forward_hook(add_forward_hook()))
		self.model(X)

		for i, hook_handle in enumerate(hook_handles):
			hook_handle.remove()

		if not self.model.training:
			self.model.train()

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
		dict_module_selector = OrderedDict()
		for module, dict_plan in self.dict_layer_output_plan.items():
			print('ADD SELECTOR -', module)
			dict_sub_selector = {}
			if module.module_type == 'activation':
				selector_cls = map_module_type_selector[module.parent.module_type]['random']
			else:
				selector_cls = map_module_type_selector[module.module_type]['random']

			if dict_plan['weight']:
				if module.module_type == 'activation':
					dict_sub_selector['weight'] = False
				else:
					dict_sub_selector['weight'] = selector_cls(tensor_size=module.weight_size)
			else:
				dict_sub_selector['weight'] = False

			if dict_plan['output']:
				dict_sub_selector['output'] = selector_cls(tensor_size=module.output_size)
			else:
				dict_sub_selector['output'] = False

			dict_module_selector[module] = dict_sub_selector

		self.dict_module_output_selector = dict_module_selector
		self.dict_layer_output_plan = dict_module_selector


class Interpreter:
	def __init__(self, model: nn.Module, layer_name_map: OrderedDict, xray_id, file_manager):
		self.model = model
		self.layer_name_map = layer_name_map.copy()
		self.dict_layer_output_plan: OrderedDict = OrderedDict()
		self.layer_output_map: OrderedDict
		self.list_layers_ordered: list = []

		self.xray_id = xray_id
		self.fm = file_manager

		self.hooked_modules = np.array([])		# todo: bunun farklı bir kullanımını oluşturmak lazım
		self.dict_output = {}

	def take_graph(self, x: torch.Tensor, batch_num: int = 0):
		layers_output = [layer for layer in self.list_layers_ordered if self.dict_layer_output_plan[layer]['output']]
		layers_weight = [layer for layer in self.list_layers_ordered if self.dict_layer_output_plan[layer]['weight']]
		# layers_grad = [layer for layer in self.list_layers_ordered if self.dict_layer_output_plan[layer]['grad']]

		d_ = self.hook_forward(
			input_tensor=x,
			modules_to_hook_outputs=layers_output,
			modules_to_hook_weights=layers_weight)

		dict_output_pruned = OrderedDict()
		for module, plan in self.dict_layer_output_plan.items():
			dict_pruned = {}

			if plan['weight']:
				tensor_pruned = plan['weight'].make_selection(d_[module]['weight'])
				self.fm.save_data(
					tensor=tensor_pruned, xray_id=self.xray_id, layer_name=module.name,
					output_type='weight', batch_num=batch_num, extension='.pt')

				dict_pruned['weight'] = tensor_pruned

			if plan['output']:
				tensor_pruned = plan['output'].make_selection(d_[module]['output'])
				self.fm.save_data(
					tensor=tensor_pruned, xray_id=self.xray_id, layer_name=module.name,
					output_type='weight', batch_num=batch_num, extension='.pt')

				dict_pruned['output'] = tensor_pruned

			dict_output_pruned[module] = dict_pruned.copy()

		return dict_output_pruned

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
					print('FORWARD HOOK -', module_name)
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

