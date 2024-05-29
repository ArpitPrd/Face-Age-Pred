import torch
import torch.nn as nn


class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.net = nn.Sequential(
		# shape = (100, 3, 256, 256)
			nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.BatchNorm2d(num_features = 16),
		# shape = (100, 16, 128, 128)
			nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.BatchNorm2d(num_features = 32),
		# shape = (100, 32, 64, 64)
			nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.BatchNorm2d(num_features = 64),
		# shape = (100, 64, 32, 32)
			nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.BatchNorm2d(num_features = 64),
		# shape = (100, 64, 16, 16)
			nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.BatchNorm2d(num_features = 64),
		# shape = (100, 64, 8, 8)
			nn.Flatten(),
		# shape = (100, 4096)
			nn.Linear(4096, 256),
		# shape = (100, 256)
			nn.ReLU(inplace = True),
			nn.Dropout(0.5),
			
			nn.Linear(256, 64),
		# shape = (100, 64)
			nn.ReLU(inplace = True),
			nn.Dropout(0.5),
			
			nn.Linear(64, 1),
		# shape = (100, 1)
			nn.ReLU(inplace = True)
		)

	def forward(self, inp):
		inp = self.net(inp)
		return inp

