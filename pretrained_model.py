from utils import save_checkpoint,save_some_examples,load_checkpoint,lab_to_rgb
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


def train_fn(disc,gen,loader,opt_disc,opt_gen,L1,bce,g_scaler,d_scaler):
    loop=tqdm(loader,leave=True)
    for data in loop:
        Ls, ab = data['L'], data['ab']
        Ls=Ls.to(config.DEVICE)
        ab=ab.to(config.DEVICE)
        
        #train Disc
        with torch.cuda.amp.autocast():
            fake_color=gen(Ls)
            # print(type(fake_color)
            fake_image = torch.cat([Ls, fake_color], dim=1)
            real_image = torch.cat([Ls, ab], dim=1)
            d_real=disc(real_image)
            d_real_loss=bce(d_real,torch.ones_like(d_real))
            d_fake=disc(fake_image.detach())
            d_fake_loss=bce(d_fake,torch.zeros_like(d_fake))
            d_loss=(d_real_loss+d_fake_loss)/2
            
        disc.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
        #Train Generator
        with torch.cuda.amp.autocast():
            fake_image = torch.cat([Ls, fake_color], dim=1)
            D_fake = disc(fake_image)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1_loss = L1(fake_color, ab) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1_loss
            # print(f"L1 Loss: {loss_meter.avg:.5f}")

        
        gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        errorGL1.append(L1_loss.cpu().detach().numpy())
        errorGTotal.append(G_loss.cpu().detach().numpy())
        errorD.append(d_loss.cpu().detach().numpy())
        np.save("./loss_Arrays/LossD",errorD)
        np.save("./loss_Arrays/LossGL1",errorGL1)
        np.save("./loss_Arrays/LossGTotal",errorGTotal)


def main():
    disc = PatchDiscriminator(in_channels=3).to(config.DEVICE)
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load("./pretrainedWeights/res18-unet.pt", map_location=config.DEVICE))
    print("Wt Loaded")
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(net_G.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

        
    train_data=Colorization(train_paths,split="train")
    train_loader= DataLoader(
        train_data,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    g_scaler=torch.cuda.amp.GradScaler()
    d_scaler=torch.cuda.amp.GradScaler()
    val_dataset = Colorization(val_paths,split="val")

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    for epoch in range(config.NUM_EPOCHS_PRE):
        print("Number of Epochs: ", epoch+1)
        train_fn(disc,net_G,train_loader,opt_disc,opt_gen,L1_LOSS, BCE,g_scaler,d_scaler)
        if (epoch+1)%1==0:
            save_some_examples(net_G, val_loader, epoch, folder="./output")
        if epoch%5==0:
            save_checkpoint(net_G, opt_gen, filename=config.CHECKPOINT_GEN_PRE)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC_PRE)

if __name__ == '__main__':
    main()
    # print(config.DEVICE)