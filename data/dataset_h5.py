import numpy as np
import pandas as pd
import pathlib, os

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			roi_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.roi_transforms = img_transforms
		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		with h5py.File(self.file_path, "r") as hdf5_file:
			dset = hdf5_file['imgs']
			for name, value in dset.attrs.items():
				print(name, value)

		print('transformations:', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		img_transforms=None):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			img_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.wsi = wsi
		self.roi_transforms = img_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('transformations: ', self.roi_transforms)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

		img = self.roi_transforms(img)
		return {'img': img, 'coord': coord}

class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]


class Custom_Dataset(Dataset):
	def __init__(self, dir_path:pathlib.Path, img_transforms=None):
		"""
		Args:
			dir_path (string): Directory path containing .jpg image patches of the slide.
			img_transforms (callable, optional): Optional transform to be applied to a sample i.e. a patch.
		"""
		self.dir_path = dir_path
		self.img_transforms = img_transforms
		self.image_files = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]
		self.length = len(self.image_files)
		self.summary()

	def __len__(self):
		return self.length

	def summary(self):
		print('Total images:', self.length)
		print('Directory:', self.dir_path)
		print('Transformations:', self.img_transforms)

	def __getitem__(self, idx):
		img_path = self.dir_path / self.image_files[idx]
		img = Image.open(img_path).convert('RGB')
	
		if self.img_transforms:
			img = self.img_transforms(img)
		else:
			img = transforms.ToTensor()(img)

		coords = np.array([int(self.image_files[idx].split('_')[-3]), 
					 		int(self.image_files[idx].split('_')[-1].split('.')[0])])
		
		return {'img': img, 'coord': coords}


