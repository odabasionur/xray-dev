import os
import json
from typing import Optional
import torch
from torch import save


class FileManager:

	def __init__(self, main_output_directory: str, xray_id: str):
		"""
		Creates the following and their .. if necessary
			./xray_outputs
			./xray_outputs/xray_id/
			./xray_outputs/xray_id/metadata
			./xray_outputs/xray_id/image
			./xray_outputs/xray_id/layer-no
			./xray_outputs/xray_id/layer-no/weight
			./xray_outputs/xray_id/layer-no/output
			./xray_outputs/xray_id/layer-no/grad

		:param main_output_directory:
		"""
		self.main_dir = main_output_directory
		self.xray_id = xray_id
		self.output_dir = os.path.join(self.main_dir, xray_id)
		self.output_image_dir = os.path.join(self.output_dir, 'image')
		self.output_metadata_dir = os.path.join(self.output_dir, 'metadata')
		self.output_metadata_path = os.path.join(self.output_metadata_dir, 'metadata.txt')
		self.non_module_folders = ['image', 'metadata']
		self.dict_module_data_dir = {}
		self.dict_module_image_dir = {}

	@staticmethod
	def create_dir_if_not_exist(directory: str):
		if not os.path.isdir(directory):
			os.makedirs(directory)

	def create_main_dir(self):
		self.create_dir_if_not_exist(directory=self.main_dir)
		self.create_dir_if_not_exist(directory=self.output_dir)

	def create_sub_folders(self, list_of_modules: list[str]):
		print('creating sun folders')
		self.create_dir_if_not_exist(self.output_image_dir)
		self.create_dir_if_not_exist(self.output_metadata_dir)
		for module_name in list_of_modules:
			print('creating sub folder for ', module_name)

			module_data_dir = os.path.join(self.output_dir, module_name.lower())
			module_image_dir = os.path.join(self.output_image_dir, module_name.lower())

			self.create_dir_if_not_exist(module_data_dir)
			self.create_dir_if_not_exist(module_image_dir)

			self.dict_module_data_dir[module_name] = module_data_dir
			self.dict_module_image_dir[module_name] = module_image_dir

		with open(self.output_metadata_path, mode='w+') as f:
			f.write('')

	def extract_output_folder_structure_by_architecture(self, list_of_modules: list):
		# todo: extractlar aynı oldu genelleştir
		self.dict_module_data_dir = {}
		self.dict_module_image_dir = {}

		for module_name in list_of_modules:
			module_data_dir = os.path.join(self.output_dir, module_name.lower())
			module_image_dir = os.path.join(self.output_image_dir, module_name.lower())

			self.dict_module_data_dir[module_name] = module_data_dir
			self.dict_module_image_dir[module_name] = module_image_dir

	def extract_output_folder_structure_by_walking(self, output_dir: str = None):
		output_dir = output_dir if isinstance(output_dir, str) else self.output_dir
		sub_folders = os.listdir(output_dir)
		self.dict_module_data_dir = {}
		self.dict_module_image_dir = {}

		module_folders = list(filter(lambda x: x not in self.non_module_folders, sub_folders))
		for module_name in module_folders:
			module_data_dir = os.path.join(self.output_dir, module_name.lower())
			module_image_dir = os.path.join(self.output_image_dir, module_name.lower())

			self.dict_module_data_dir[module_name] = module_data_dir
			self.dict_module_image_dir[module_name] = module_image_dir

	def get_data_filename(self, xray_id: str, layer_name, output_type: str, batch_num: int, extension='.pt'):
		return f'{xray_id}-{layer_name}-{output_type}-{batch_num}{extension}'

	def get_data_path(self, xray_id: str, layer_name: str, output_type: str, batch_num: int, extension='.pt'):
		data_filename = self.get_data_filename(
			xray_id=xray_id, layer_name=layer_name, output_type=output_type, batch_num=batch_num, extension=extension
		)
		module_dir = self.dict_module_data_dir.get(layer_name, self.output_dir)
		data_path = os.path.join(module_dir, data_filename)
		return data_path

	def get_image_path(self, xray_id: str, layer_name: str, output_type: str, batch_num: int, extension='.pt'):
		data_filename = self.get_data_filename(
			xray_id=xray_id, layer_name=layer_name, output_type=output_type, batch_num=batch_num,
			extension=extension
		)
		image_dir = self.dict_module_image_dir.get(layer_name, self.output_dir)
		data_path = os.path.join(image_dir, data_filename)
		return data_path

	def get_data_json(self, xray_id: str, layer_name: str, output_type: str, batch_num: int, extension='.pt'):
		data_path = self.get_data_path(
			xray_id=xray_id, layer_name=layer_name, output_type=output_type, batch_num=batch_num, extension=extension)

		return {
			'xray_id': xray_id,
			'layer_name': layer_name,
			'output_type': output_type,
			'batch_num': batch_num,
			'data_path': data_path,
		}

	def update_metadata(self, metadata: dict):
		with open(self.output_metadata_path, mode='a+') as f:
			metadata_str = json.dumps(metadata)
			f.write(metadata_str + '\n')

	@staticmethod
	def read_metadata(path: str):
		with open(path) as f:
			metadata_text = f.read()
		return [json.loads(metadata_line) for metadata_line in metadata_text.split('\n')[:-1]]

	def save_data(
			self,
			tensor: torch.Tensor,
			xray_id: str,
			layer_name: str,
			output_type: str,
			batch_num: int,
			extension='.pt'):

		json_metadata = self.get_data_json(
			xray_id=xray_id, layer_name=layer_name, output_type=output_type, batch_num=batch_num, extension=extension)
		save(tensor, json_metadata['data_path'])
		self.update_metadata(metadata=json_metadata)

	def save_plot(
			self,
			plot,
			xray_id: str,
			layer_name: str,
			output_type: str,
			batch_num: int,
			extension='.png'):

		image_path = self.get_image_path(
			xray_id=xray_id, layer_name=layer_name, output_type=output_type, batch_num=batch_num, extension=extension)
		plot.savefig(image_path)

	def save_history(self, loss_history, accuracy_history):
		import pandas as pd
		import matplotlib.pyplot as plt
		import seaborn as sns

		plt.cla()
		plt.clf()

		sns.lineplot(y=accuracy_history, x=list(range(len(accuracy_history)))).set_title('Accuracy History')
		plt.savefig(os.path.join(self.output_metadata_dir, 'accuracy_plot.png'))

		sns.lineplot(y=loss_history, x=list(range(len(loss_history)))).set_title(f'Loss History')
		plt.ylim(-0.01, 1)
		plt.savefig(os.path.join(self.output_metadata_dir, 'loss_plot.png'))

		df_hist = pd.DataFrame([loss_history, accuracy_history], index=['loss', 'accuracy']).T
		df_hist.to_csv(os.path.join(self.output_metadata_dir, 'history.csv'), index=False)

