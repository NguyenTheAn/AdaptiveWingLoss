import torch
import shutil
import numpy as np
from PIL import Image
import os
import cv2
from core import models
from tqdm import tqdm

PRETRAINED_WEIGHTS = "ckpt/WFLW_4HG.pth"
GRAY_SCALE = False
HG_BLOCKS = 4
END_RELU = False
NUM_LANDMARKS = 98

list_images = []
with open("../match_data/train_p01_s02.txt", 'r') as file:
    lines = file.readlines()
    for line in lines:
        list_images.append(line)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)
checkpoint = torch.load(PRETRAINED_WEIGHTS)
if 'state_dict' not in checkpoint:
    model_ft.load_state_dict(checkpoint)
else:
    pretrained_weights = checkpoint['state_dict']
    model_weights = model_ft.state_dict()
    pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                            if k in model_weights}
    model_weights.update(pretrained_weights)
    model_ft.load_state_dict(model_weights)

model_ft = model_ft.to(device)

for path in tqdm(list_images):
    name = path.split("/")[-1]
    p = path.split("/")[-2]
    image = cv2.imread("../match_data/"+p+"/"+name)
    image = cv2.resize(image, (256, 256))
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis = 0)
    image = torch.from_numpy(image).float().div(255.0)
    image = image.to(device)
    with torch.no_grad():
        outputs, _ = model_ft(image)
        pred_heatmap = outputs[-1][:, :-1, :, :][0].detach().cpu()
        if not os.path.isdir("/mnt/vinai/match_data/train/"):
            os.makedirs("/mnt/vinai/match_data/train/")
        np.save("/mnt/vinai/match_data/train/" + name[:-4], pred_heatmap)

list_images = []
with open("../match_data/valid_p01_s02.txt", 'r') as file:
    lines = file.readlines()
    for line in lines:
        list_images.append(line)

for path in tqdm(list_images):
    name = path.split("/")[-1]
    image = cv2.imread(path)
    image = cv2.resize(image, (256, 256))
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis = 0)
    image = torch.from_numpy(image).float().div(255.0)
    image = image.to(device)
    with torch.no_grad():
        outputs, _ = model_ft(image)
        pred_heatmap = outputs[-1][:, :-1, :, :][0].detach().cpu()
        if not os.path.isdir("/mnt/vinai/match_data/valid/"):
            os.makedirs("/mnt/vinai/match_data/valid/")
        np.save("/mnt/vinai/match_data/valid/" + name[:-4], pred_heatmap)