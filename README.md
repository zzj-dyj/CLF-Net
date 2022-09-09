# CLF-Net  
The code of "CLF-Net: "Contrastive Learning for Infrared and Visible Image Fusion Network "

## Environmental Requirements  
python=3.7;   
pytorch=1.5.1 

  
## Dataset  
The training dataset is placed in './datasets/training_img/'.  
Making sure the name of images is exactly the same in different sensor folders.  
In the data preparation stage, the training image needs to be cropped into 128*128, the image cropping code ` img_crop.py ` .



## Parameter Setting  
  
```python
# ./config/CLF_Net.yaml

PROJECT: 
  name: 'CLF_Net_Image_Fusion' # Project name
  save_path: './work_dirs/' # Project save path, the training model will be saved to this path

TRAIN_DATASET: 
  root_dir: './datasets/cropped_img/' # The root directory of the training dataset
  sensors: [ 'Vis', 'Inf' ] # The type of data that the training dataset contains
  channels: 1 # Number of channels of images in the training data

TRAIN: 
  batch_size: 4 
  max_epoch: 20 
  gpu_id: 0
  val_interval: 1 
  resume: None # Loading weight path used to continue training

TEST_DATASET: 
  root_dir: './datasets/test_img/' # The root directory of the test dataset
  sensors: [ 'Vis', 'Inf' ] # The type of data that the testing dataset contains
  channels: 1 # Number of channels of images in the training data

TEST: 
  batch_size: 1
  weight_path: './work_dirs/CLF_Net/CLF_Net.pth' # The weight path for testing
  save_path: './results' # path of the results

MODEL: # 
  model_name: 'CLF_Net' 
  input_channels: 1 
  out_channels: 16 
  input_sensors: [ 'Vis', 'Inf' ] # The data type of input
  coder_layers: 4 
  decoder_layers: 4 

```  

## Training And Testing  
  
### Training  
Change the default value of '--train' to True. Run  ` run.py`  for training.

```python
# ./run.py line 14

def get_args():
    parser = argparse.ArgumentParser(description='run')

    parser.add_argument('--config', type=str, default='./config/CLF_Net.yaml')
    parser.add_argument('--train', default=True)
    parser.add_argument('--test', default=False)
    args = parser.parse_args()
    return args
```
  
### Testing  
Change the default value of '--test' to True. Run  ` run.py`  for testing  

```python
# ./run.py line 14

def get_args():
    parser = argparse.ArgumentParser(description='run')

    parser.add_argument('--config', type=str, default='./config/CLF_Net.yaml')
    parser.add_argument('--train', default=False)
    parser.add_argument('--test', default=True)
    args = parser.parse_args()
    return args
``` 
    
### Weight 
The test weight in the paper is placed at './work_dirs/CLF_Net/CLF_Net.pth'
 

# CLF_Net
