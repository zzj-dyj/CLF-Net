import os
import torch
import sys
sys.path.append('../')
from tqdm import tqdm
from core.loss import *
from core.util import debug
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


def train(model, train_datasets, configs):
	if not os.path.exists(os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name'])):
		os.mkdir(os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name']))

	model.train()

	# train_writer = SummaryWriter(log_dir=os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name']))
	print(f'Run Tensorboard:\n tensorboard --logdir=' + configs['PROJECT']['save_path'] + '/' + configs['PROJECT'][
		'name'] + '/')

	if configs['TRAIN']['resume'] == 'None':
		start_epoch = 1
	else:
		start_epoch = torch.load(configs['TRAIN']['resume'])['epoch'] + 1

	is_use_gpu = torch.cuda.is_available()


	train_dataloader = DataLoader(train_datasets, batch_size=configs['TRAIN']['batch_size'], shuffle=True)
	train_num_iter = len(train_dataloader)

	all_iter = 0
	for epoch in range(start_epoch, configs['TRAIN']['max_epoch'] + 1):

		loss_epoch = 0

		with tqdm(total=train_num_iter) as train_bar:
			for iter, data in enumerate(train_dataloader):

				if is_use_gpu:
					model = model.cuda(configs['TRAIN']['gpu_id'])
					vis_img = data['Vis'].cuda(configs['TRAIN']['gpu_id'])
					inf_img = data['Inf'].cuda(configs['TRAIN']['gpu_id'])
					#data = {sensor: data[sensor].cuda(configs['TRAIN']['gpu_id']) for sensor in data}

				if epoch-1 == 0 and iter == 0:
					model.data_dependent_initialize(vis_img, inf_img)
					model.setup()
				model.optimize_parameters(vis_img, inf_img)

				loss = model.calculate_loss(vis_img, inf_img)

				loss_epoch += loss.item()

				train_bar.set_description(
					'Epoch: {}/{}. TRAIN. Iter: {}/{}. All loss: {:.5f}'.format(
						epoch, configs['TRAIN']['max_epoch'], iter + 1, train_num_iter,
						                                      loss_epoch / train_num_iter))

				all_iter += 1
				train_bar.update(1)

			model.update_learning_rate()

			if configs['TRAIN']['val_interval'] is not None and all_iter % configs['TRAIN']['val_interval'] == 0:
				torch.save({'model': model, 'epoch': epoch},
				           os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name'],
				                        f'model_{epoch}.pth'))
