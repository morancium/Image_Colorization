from utils import save_checkpoint,save_some_examples,load_checkpoint,lab_to_rgb
import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import config
from dataset import Colorization
from PatchDiscriminator_model import PatchDiscriminator
from generator_model import Generator
import glob
from torch.utils.data import Dataset,DataLoader
from fastai.data.external import untar_data, URLs

coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"
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
        # print("d_loss: ",d_loss)
        
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
        # print("L1_loss: ",L1_loss)

def main():
    disc = PatchDiscriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=1, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    train_data = Colorization(train_paths, split="train")
    train_loader = DataLoader(
        train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = Colorization(val_paths, split="val")

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    for epoch in range(config.NUM_EPOCHS):
        print("Number of Epochs: ", epoch + 1)
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler)
        
        if (epoch + 1) % 1 == 0:
            save_some_examples(gen, val_loader, epoch, folder="./evaluations")
        if epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)


if __name__ == '__main__':
    main()