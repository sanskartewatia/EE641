import train.utils_caps as utils
from train.capsules_ucf101 import CapsNet
from skvideo.io import vread, vwrite
from torch.utils.data import DataLoader
import torch
import cv2
import numpy as np

from train.load_ucf101_pytorch_gaus import UCF101DataLoader 

model_file_path = './trained/active_learning/checkpoints_ucf101_capsules_i3d/i3d-ucf101-5e-05_8_ADAM_capsules_Multi_100perRand_GausInterp_Dropout_PreTrainedCharades_RGB_Spread_BCE_epoch_076.pth'
    
model = CapsNet()
model.load_previous_weights(model_file_path)
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

vid_size = [224, 224]
validationset = UCF101DataLoader('validation', vid_size, 1, use_random_start_frame=False)

val_data_loader = DataLoader(
    dataset=validationset,
    batch_size=1,
    num_workers=32,
    shuffle=False
)

for minibatch in val_data_loader:
    data = minibatch['data'].float()
    action = minibatch['action'].float()
    data = data.to(device)
    action = action.to(device)
    output, predicted_action = model(data, action)
    maskout = output.cpu()
    maskout_np = maskout.data.numpy()
    # print(maskout_np)

    # use threshold to make mask binary
    # maskout_np[maskout_np > 0] = 1
    # maskout_np[maskout_np < 0] = 0
    maskout_np[maskout_np > -5] = 1
    maskout_np[maskout_np < -5] = 0

    frames = maskout_np[0][0] * 255
    frames = frames.astype(np.uint8)
    
    vwrite('KNNv_Surfing_g07_c01.mp4', frames)
    