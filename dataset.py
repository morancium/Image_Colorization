import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import config
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
import numpy as np



class Colorization(Dataset):
    def __init__(self, paths, split="train"):
        # self.root_dir = path
        if split == "train":
            self.transforms = transforms.Compose(
                [
                    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),  
                ]
            )
        elif split == "val":
            self.transforms = transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BICUBIC)

        self.split = split
        self.size = config.IMAGE_SIZE
        self.list_files = paths

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_file = Image.open(img_file).convert("RGB")
        img_file = self.transforms(img_file)
        img_file = np.array(img_file)
        img_lab = rgb2lab(img_file).astype("float32")  # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50.0 - 1.0  # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110.0  # Between -1 and 1

        return {"L": L, "ab": ab}
        # return  L, ab


# if __name__ == "__main__":
#     dataset = Colorization(train_paths)
#     loader = DataLoader(dataset, batch_size=16)
#     # print(loader)
#     data = next(iter(loader))
#     Ls, abs_ = data["L"], data["ab"]
#     # print(Ls.shape, abs_.shape)
#     save_image(Ls, "./x.png")
#     save_image(abs_, "./y.png")
#     # save_image(abs_[:,0], "/z.png")
#     # print(len(loader))
