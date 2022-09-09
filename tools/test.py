# coding: utf-8
import imageio
import torch
import os
import glob
import time

import numpy as np

import scipy.misc
import scipy.ndimage

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data2(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.tif"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.extend(glob.glob(os.path.join(data_dir, "*.png")))
    data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
    data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    return data


def input_setup2(data_vi, data_ir, index):

    sub_ir_sequence = []
    sub_vi_sequence = []
    _ir = imread(data_ir[index])
    _vi = imread(data_vi[index])

    input_ir = (_ir - 127.5) / 127.5
    w, h = input_ir.shape
    input_ir = input_ir.reshape([w, h, 1])
    input_vi = (_vi - 127.5) / 127.5
    w, h = input_vi.shape
    input_vi = input_vi.reshape([w, h, 1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir = np.asarray(sub_ir_sequence)
    train_data_vi = np.asarray(sub_vi_sequence)
    return train_data_ir, train_data_vi



def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
    return scipy.misc.imsave(path, image)


def test(model, configs, load_weight_path=False):
    data_ir = prepare_data2(os.path.join(configs['TEST_DATASET']['root_dir'], 'Inf'))
    data_vi = prepare_data2(os.path.join(configs['TEST_DATASET']['root_dir'], 'Vis'))

    # data_ir = prepare_data2(os.path.join('./datasets/road', 'Inf'))
    # data_vi = prepare_data2(os.path.join('./datasets/road', 'Vis'))
    model.eval()
    if load_weight_path:
        assert configs['TEST']['weight_path'] != 'None', 'Test Need To Resume Chekpoint'
        weight_path = configs['TEST']['weight_path']
        checkpoint = torch.load(weight_path)

        model.load_state_dict(checkpoint['model'].state_dict(), strict=False)

    model = model.cuda()
    path = configs['TEST']['save_path']
    if not os.path.exists(path):
        os.makedirs(path)


    with torch.no_grad():
        total_time = []

        for i in range(len(data_ir)):

            train_data_ir, train_data_vi = input_setup2(data_vi, data_ir, i)

            train_data_ir = train_data_ir.transpose([0, 3, 1, 2])
            train_data_vi = train_data_vi.transpose([0, 3, 1, 2])

            train_data_ir = torch.tensor(train_data_ir).float().to(device)
            train_data_vi = torch.tensor(train_data_vi).float().to(device)

            data_test_vi = train_data_vi.cuda()
            data_test_ir = train_data_ir.cuda()

            start = time.time()
            result = model(data_test_vi, data_test_ir)

            result = np.squeeze(result.cpu().numpy() * 127.5 + 127.5)

            if i < 9:
                save_path = os.path.join(path, '00'+str(i + 1) + ".jpg")
            else:
                save_path = os.path.join(path, '0'+str(i + 1) + ".jpg")

            end = time.time()
            imsave(result, save_path)
            print("Testing [%d] success,Testing time is [%f]" % (i, end - start))
            total_time.append(end - start)
            pass
        a = np.array(total_time)
        print('Mean of time:', np.average(a))
        print('Std deviation of time:', np.sqrt(np.var(a)))

# if __name__ == '__main__':
#     test_all()
