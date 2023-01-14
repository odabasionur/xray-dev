from collections import OrderedDict
from typing import Optional
import graphviz
import torch
from torch import nn


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


class TraceArchitecture:
	def __init__(
			self,
			model: Optional[nn.Module] = None,
			input_size: Optional[tuple] = None,
			output: Optional[torch.Tensor] = None):

		self.model = model
		self.input_size = input_size
		if isinstance(output, torch.Tensor):
			self.output = output
		else:
			if isinstance(self.model, nn.Module) & isinstance(self.input_size, tuple):
				output = self.model(torch.ones(*self.input_size))
				self.output = output

		root_grad = DataGradFn(self.output.grad_fn)
		root_grad.root = True

		self.dict_graph_module = OrderedDict()
		self.dict_graph_module_filtered = OrderedDict()

	@staticmethod
	def _add_node_if_not_exist(dict_modify, key):
		if dict_modify.get(key) is None:
			dict_modify[key] = []

	@staticmethod
	def is_keyword_necessary(keyword):
		key = keyword.name() if keyword else keyword
		print('is_keyword_necessary', key)
		non_layer_variables = ['addbackward', 'unsqueezebackward', 'squeezebackward', 'accumulategrad']
		if isinstance(key, str):
			# If this if statement need to be change in the future, it must be
			# considered that the condition of `keyword == None` is possible
			for non_layer_var in non_layer_variables:
				if non_layer_var in key.lower():
					return False
		else:
			print('else')
			return False
		return True

	def travel_through_grad_fn(self, grad_fn, child_fn):
		"""
		Since torch.Variable.grad_fn.next_functions keep functions family
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
		q = key.name() if key else key
		w = child_key.name() if child_key else child_key
		print(key, child_key)

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

	def customize_graph_of_grad_fn(self):
		pass

	def get_architecture_in_detail(self):
		self.travel_through_grad_fn(grad_fn=self.output.grad_fn, child_fn=None)
		return self.dict_graph_module

	def get_architecture_core(self):
		if len(self.dict_graph_module) == 0:
			self.get_architecture_in_detail()

		self.filter_graph_dict(key=self.output.grad_fn, child_key=None)


def plot_graph_from_dict(dict_graph, view=True):
    diagraph = graphviz.Digraph(comment='Example Graph')
    diagraph.graph_attr.update({'rankdir': 'TB'})
    for node in dict_graph.keys():
        diagraph.node(node, label=node.split('->')[0])

    for node, sub_nodes in dict_graph.items():
        print('1', node, sub_nodes)
        for sub_node in sub_nodes:
            print('  2', sub_node)
            diagraph.edge(sub_node, node)
            diagraph.edge(node, sub_node, color='red', fontcolor='red', fontsize="8")
            # diagraph.edge(node, sub_node, label='3.45', color='red', fontcolor='red', fontsize="8")

    # diagraph.render(r'D:\\Onur\\Projects\\computation_with_dag\\example_graph.gv', view=view)
    diagraph.view(r'D:\\Onur\\Projects\\computation_with_dag\\example_graph.gv')



if __name__ == '__main__':
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

	dgf = DataGradFn(output.grad_fn)
	print(dgf)

	print('başlıyor')
	ta = TraceArchitecture(output=output)

	ta.get_architecture_in_detail()
	print(ta.dict_graph_module)

	ta.get_architecture_core()
	print(ta.dict_graph_module_filtered)
	print(ta.dict_graph_module)

