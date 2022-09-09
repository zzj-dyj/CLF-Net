import torch
import argparse
import sys
sys.path.append('../')
from core.model import *
from tools import *
from core.dataset import Fusion_Datasets
import torchvision.transforms as transforms
from core.util import load_config, count_parameters
import warnings

warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description='run')

    parser.add_argument('--config', type=str, default='./config/CLF_Net.yaml')
    parser.add_argument('--train',  default=True)
    parser.add_argument('--test',  default=False)
    args = parser.parse_args()
    return args


def runner(args):
    configs = load_config(args.config)
    # project_configs = configs['PROJECT']
    model_configs = configs['MODEL']
    train_configs = configs['TRAIN']
    # test_configs = configs['TEST']
    train_dataset_configs = configs['TRAIN_DATASET']
    test_dataset_configs = configs['TEST_DATASET']
    # input_size = train_dataset_configs['input_size'] if args.train else test_dataset_configs['input_size']

    if train_dataset_configs['channels'] == 3:
        base_transforms = transforms.Compose(
            [transforms.ToTensor()])

    elif train_dataset_configs['channels'] == 1:
        base_transforms = transforms.Compose(
            [transforms.ToTensor()])  # ,


    train_datasets = Fusion_Datasets(train_dataset_configs, base_transforms)
    test_datasets = Fusion_Datasets(test_dataset_configs, base_transforms, True)

    model = eval(model_configs['model_name'])(model_configs)
    print('Model Para:', count_parameters(model))

    if train_configs['resume'] != 'None':

        checkpoint = torch.load(train_configs['resume'])
        model.load_state_dict(checkpoint['model'].state_dict())


    if args.train:
        train(model,train_datasets, configs)
    if args.test:
        test(model, configs, load_weight_path=True)


if __name__ == '__main__':
    args = get_args()
    runner(args)
