import torch
import config
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab, lab2rgb

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def save_some_examples(gen, val_loader, epoch, folder):
    data = next(iter(val_loader))
    Ls, abs_ = data["L"], data["ab"]
    Ls = Ls.to(config.DEVICE)
    abs_ = abs_.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        fake_color = gen(Ls)
        fake_imgs = lab_to_rgb(Ls, fake_color)
        # print(type(Ls),type(fake_color))
        real_imgs = lab_to_rgb(Ls, abs_)
        # print(Ls.shape)
        fig = plt.figure(figsize=(15, 8))
        for i in range(5):
            ax = plt.subplot(3, 5, i + 1)
            ax.imshow(Ls[i][0].cpu(), cmap="gray")
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 5)
            ax.imshow(fake_imgs[i])
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 10)
            ax.imshow(real_imgs[i])
            ax.axis("off")
        # plt.show()
        fig.savefig(folder+f"/Epoch_{epoch+1}.png")
    gen.train()

def saveSingleExample(gen, val_loader, epoch, folder):
    data = next(iter(val_loader))
    Ls, abs_ = data["L"], data["ab"]
    Ls = Ls.to(config.DEVICE)
    abs_ = abs_.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        fake_color = gen(Ls)
        fake_imgs = lab_to_rgb(Ls, fake_color)
        # print(type(Ls),type(fake_color))
        real_imgs = lab_to_rgb(Ls, abs_)
        # print(Ls.shape)
        fig = plt.figure(figsize=(6,2))
        for i in range(1):
            ax = plt.subplot(1, 3, 1)
            ax.set_title('Black and White')
            ax.imshow(Ls[i][0].cpu(), cmap="gray")
            ax.axis("off")
            ax = plt.subplot(1, 3, 2)
            ax.set_title('Coloured')
            ax.imshow(fake_imgs[i])
            ax.axis("off")
            ax = plt.subplot(1, 3, 3)
            ax.set_title('Ground Truth')
            ax.imshow(real_imgs[i])
            ax.axis("off")
        # plt.show()
        fig.savefig(folder+f"/Epoch_{epoch+1}.png")
    gen.train()

def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    # print(type(L),type(ab))

    L = (L + 1.0) * 50.0
    ab = ab * 110.0
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    rgb = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)
