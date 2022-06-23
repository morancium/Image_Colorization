from utils import save_checkpoint,save_some_examples,load_checkpoint,lab_to_rgb, saveSingleExample
import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import config
from dataset import Colorization
from PatchDiscriminator_model import PatchDiscriminator
from pretrained_generator import build_res_unet
from fastai.data.external import untar_data, URLs
import glob
from torch.utils.data import Dataset,DataLoader
from fastai.data.external import untar_data, URLs

errorGL1=[]
errorGTotal=[]
errorD=[]
coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"
use_colab = True

if use_colab == True:
    path = coco_path

paths = glob.glob(path + "/*.jpg")  # Grabbing all the image file names
np.random.seed(123)
paths_subset = np.random.choice(
    paths, 10_000, replace=False
)  # choosing 1000 images randomly
rand_idxs = np.random.permutation(10_000)
train_idxs = rand_idxs[:8000]  # choosing the first 8000 as training set
val_idxs = rand_idxs[8000:]  # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]



def main():
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    opt_gen = optim.Adam(net_G.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    disc = PatchDiscriminator(in_channels=3).to(config.DEVICE)
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    # net_G.load_state_dict(torch.load("./pretrainedWeights/res18-unet.pt", map_location=config.DEVICE))
    load_checkpoint("./output/_gen.pth.tar", net_G, opt_gen, lr=2e-4)
    print("Wt Loaded")
    val_dataset = Colorization(val_paths,split="val")

    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=True)
    # save_some_examples(net_G, val_loader,0, folder="./test")
    # for i in range(10):
    #     saveSingleExample(net_G, val_loader, i, folder="./test")

if __name__ == '__main__':
    main()
    # print(config.DEVICE)