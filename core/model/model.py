import torch
import torch.nn as nn


class Residual_Block(nn.Module):
	def __init__(self, i_channel, o_channel, identity=None, end=False):
		super(Residual_Block, self).__init__()

		self.in_channels = i_channel
		self.out_channels = o_channel
		self.conv1 = nn.Conv2d(in_channels=self.in_channels,
							   out_channels=self.in_channels,
							   kernel_size=1,
							   stride=1,
							   padding=0,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(self.in_channels)


		self.conv2 = nn.Conv2d(in_channels=self.in_channels,
							   out_channels=self.out_channels,
							   kernel_size=3,
							   stride=1,
							   padding=1,
							   bias=False)
		self.bn2 = nn.BatchNorm2d(self.out_channels)

		# self.conv3 = nn.Conv2d(in_channels=self.in_channels,
		# 					   out_channels=self.out_channels ,
		# 					   kernel_size=3,
		# 					   stride=1,
		# 					   padding=1,
		# 					   bias=False)
		# self.bn3 = nn.BatchNorm2d(self.out_channels)

		self.identity_block = nn.Conv2d(in_channels=self.in_channels,
				  						out_channels=self.out_channels ,
				  						kernel_size=1,
				 						stride=1,
				  						padding=0,
				  						bias=False)
		self.identity = identity
		self.end = end
		self.tanh = nn.Tanh()
		self.lrelu = nn.LeakyReLU(inplace=True)

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.lrelu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		# 将单元的输入直接与单元输出加在一起
		if self.identity:
			residual = self.identity_block(x)
		out += residual
		if self.end:
			out = self.tanh(out)
		else:
			out = self.lrelu(out)
		return out


class encoder(nn.Module):
	"""docstring for dense"""

	def __init__(self, in_channels, out_channels):
		super(encoder, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels

		# self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=5, stride=1, padding=2)
		# self.bn = nn.BatchNorm2d(self.out_channels)
		# self.lrelu = nn.LeakyReLU(inplace=True)
		self.res_block0 = Residual_Block(self.in_channels, self.out_channels, identity=True)
		self.res_block1 = Residual_Block(self.out_channels, self.out_channels, identity=False)
		self.res_block2 = Residual_Block(self.out_channels, self.out_channels * 2, identity=True)
		self.res_block3 = Residual_Block(self.out_channels * 2, self.out_channels * 4, identity=True)


	def forward(self, x):
		# feat = self.conv1(x)
		# feat = self.bn(feat)
		# feat = self.lrelu(feat)
		feat = self.res_block0(x)
		feat = self.res_block1(feat)
		feat = self.res_block2(feat)
		feat = self.res_block3(feat)
		return feat

class decoder(nn.Module):
	"""docstring for dense"""

	def __init__(self, in_channels, out_channels):
		super(decoder, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.res_block1 = Residual_Block(self.in_channels * 8, self.in_channels * 4, identity=True)
		self.res_block2 = Residual_Block(self.in_channels * 4, self.in_channels * 2, identity=True)
		self.res_block3 = Residual_Block(self.in_channels * 2, self.in_channels, identity=True)
		self.res_block4 = Residual_Block(self.in_channels, self.out_channels, identity=True, end=True)


	def forward(self, x):

		feat = self.res_block1(x)
		feat = self.res_block2(feat)
		feat = self.res_block3(feat)
		feat = self.res_block4(feat)

		return feat

class Generator(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(Generator, self).__init__()
		self.vis_encoder = encoder(in_channels, out_channels)
		self.inf_encoder = encoder(in_channels, out_channels)
		self.decoder = decoder(out_channels, in_channels)


	def forward(self, vis_img, inf_img, only_encoder=False):
		feat_vis = self.vis_encoder(vis_img)
		feat_inf = self.inf_encoder(inf_img)
		feat = torch.cat([feat_vis, feat_inf], dim=1)
		# feat = feat_vis + feat_inf
		if not only_encoder:
			feat = self.decoder(feat)
		return [feat_vis, feat_inf, feat]



